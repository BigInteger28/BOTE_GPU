package main

import (
	"bufio"
	"container/heap"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
	"runtime"
	"C" // voor cgo
)

/*
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -L. -lsimulate -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart
#cgo CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
#include "simulate.h"

// Declareer de functies zonder attributen
void simulateDepthVsDepthCUDA(const char* generatedEngines, int numGenerated,
                              const char* depthInputEngines, int numDepth, int* scoreDiffs);
void simulateDepthVsFixedCUDA(const char* generatedEngines, int numGenerated,
                              const char* fixedInputEngines, int numFixed, int* scoreDiffs);
*/
import "C"

// **Globale definities**
type ElementDepth struct {
	depths [4]byte // Dieptes 1 tot 4
}

var progressMutex sync.Mutex
var progressComparisons int64
var progress int32
var elementsDepthArray [256]ElementDepth
var moveToIndexArray [256]int
var depthToElement = [5]byte{'W', 'V', 'A', 'L', 'D'}
var moveWins = [5][5]uint8{
	{0, 1, 0, 2, 0}, // W vs W,V,A,L,D
	{2, 0, 1, 0, 0}, // V
	{0, 2, 0, 1, 0}, // A
	{1, 0, 2, 0, 0}, // L
	{0, 0, 0, 0, 0}, // D
}

func init() {
	elementsDepthArray['W'] = ElementDepth{[4]byte{'L', 'A', 'V', 'W'}}
	elementsDepthArray['V'] = ElementDepth{[4]byte{'W', 'L', 'A', 'V'}}
	elementsDepthArray['A'] = ElementDepth{[4]byte{'V', 'W', 'L', 'A'}}
	elementsDepthArray['L'] = ElementDepth{[4]byte{'A', 'V', 'W', 'L'}}
	moveToIndexArray['W'] = 0
	moveToIndexArray['V'] = 1
	moveToIndexArray['A'] = 2
	moveToIndexArray['L'] = 3
	moveToIndexArray['D'] = 4
}

type Player struct {
	available [5]int
	moves     [13]byte
	moveCount int
}

type engineResult struct {
	engine string
	score  int
}

type minHeap []engineResult

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(engineResult)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// **Hulpfuncties**
func getElementFromCode(depth int) byte {
	if depth < 1 || depth > 5 {
		return 0
	}
	return depthToElement[depth-1]
}

func getElementByDepth(prevElement byte, depth int) byte {
	if depth == 5 {
		return 'D'
	}
	if prevElement == 0 {
		return 0
	}
	if prevElement == 'D' {
		prevElement = 'L'
	}
	if depth < 1 || depth > 4 {
		return 0
	}
	return elementsDepthArray[prevElement].depths[depth-1]
}

func chooseAvailableElement(target byte, available *[5]int) byte {
	targetIdx := moveToIndexArray[target]
	if available[targetIdx] > 0 {
		return target
	}
	current := target
	for i := 0; i < 5; i++ {
		current = elementsDepthArray[current].depths[0]
		currentIdx := moveToIndexArray[current]
		if available[currentIdx] > 0 {
			return current
		}
	}
	if available[4] > 0 {
		return 'D'
	}
	return 0
}

func getLastElement(available *[5]int) byte {
	candidates := [5]byte{'W', 'V', 'A', 'L', 'D'}
	for _, c := range candidates {
		if available[moveToIndexArray[c]] > 0 {
			return c
		}
	}
	return 0
}

func determineWinner(move1, move2 byte) int {
	move1Idx := moveToIndexArray[move1]
	move2Idx := moveToIndexArray[move2]
	if move1Idx < 0 || move2Idx < 0 {
		return 0
	}
	return int(moveWins[move1Idx][move2Idx])
}

// **Simulatiefuncties**
func simulateDepthGame(engine1, engine2 string) (p1Score, p2Score int) {
	if len(engine1) != 12 || len(engine2) != 12 {
		return -1, -1
	}
	var p1, p2 Player
	p1.available = [5]int{3, 3, 3, 3, 1}
	p2.available = [5]int{3, 3, 3, 3, 1}
	for i := 0; i < 12; i++ {
		depth1 := int(engine1[i] - '0')
		depth2 := int(engine2[i] - '0')
		var move1, move2 byte
		if i == 0 {
			move1 = chooseAvailableElement(getElementFromCode(depth1), &p1.available)
			move2 = chooseAvailableElement(getElementFromCode(depth2), &p2.available)
		} else {
			move1 = chooseAvailableElement(getElementByDepth(p2.moves[i-1], depth1), &p1.available)
			move2 = chooseAvailableElement(getElementByDepth(p1.moves[i-1], depth2), &p2.available)
		}
		if move1 == 0 || move2 == 0 {
			return -1, -1
		}
		p1.available[moveToIndexArray[move1]]--
		p1.moves[p1.moveCount] = move1
		p1.moveCount++
		p2.available[moveToIndexArray[move2]]--
		p2.moves[p2.moveCount] = move2
		p2.moveCount++
		winner := determineWinner(move1, move2)
		if winner == 1 {
			p1Score++
		} else if winner == 2 {
			p2Score++
		}
	}
	move1 := getLastElement(&p1.available)
	move2 := getLastElement(&p2.available)
	if move1 != 0 {
		p1.available[moveToIndexArray[move1]]--
		p1.moves[p1.moveCount] = move1
		p1.moveCount++
	}
	if move2 != 0 {
		p2.available[moveToIndexArray[move2]]--
		p2.moves[p2.moveCount] = move2
		p2.moveCount++
	}
	winner := determineWinner(move1, move2)
	if winner == 1 {
		p1Score++
	} else if winner == 2 {
		p2Score++
	}
	return p1Score, p2Score
}

func simulateFixedGame(engine1, engine2 string) (p1Score, p2Score int) {
	if len(engine1) != 13 || len(engine2) != 13 {
		return -1, -1
	}
	for i := 0; i < 13; i++ {
		move1, move2 := engine1[i], engine2[i]
		validMoves := map[byte]bool{'W': true, 'V': true, 'A': true, 'L': true, 'D': true}
		if !validMoves[move1] || !validMoves[move2] {
			return -1, -1
		}
		winner := determineWinner(move1, move2)
		if winner == 1 {
			p1Score++
		} else if winner == 2 {
			p2Score++
		}
	}
	return p1Score, p2Score
}

// **Engine-generatie**
func generateEngines(startDepth string) []string {
	var engines []string
	remainingLength := 12 - len(startDepth)
	hasFive := strings.Contains(startDepth, "5")
	if remainingLength < 0 {
		return engines
	}
	if startDepth != "" {
		for _, digit := range startDepth {
			if digit < '1' || digit > '5' {
				return engines
			}
		}
		generateRemaining(startDepth, remainingLength, hasFive, &engines)
	} else {
		for firstDigit := '1'; firstDigit <= '5'; firstDigit++ {
			prefix := string(firstDigit)
			hasFiveLocal := firstDigit == '5'
			generateRemaining(prefix, 11, hasFiveLocal, &engines)
		}
	}
	return engines
}

func generateRemaining(prefix string, remainingLength int, hasUsedFive bool, engines *[]string) {
	if remainingLength == 0 {
		if len(prefix) == 12 {
			*engines = append(*engines, prefix)
		}
		return
	}
	for digit := '1'; digit <= '5'; digit++ {
		if digit == '5' && hasUsedFive {
			continue
		}
		newPrefix := prefix + string(digit)
		generateRemaining(newPrefix, remainingLength-1, hasUsedFive || digit == '5', engines)
	}
}

// **GPU-evaluatie**
func evaluateBatchGPU(engines []string, depthInputEngines []string, fixedInputEngines []string) []int {
	numEngines := len(engines)
	numDepth := len(depthInputEngines)
	numFixed := len(fixedInputEngines)

	enginesBytes := make([]byte, numEngines*12)
	for i, engine := range engines {
		copy(enginesBytes[i*12:], engine)
	}

	depthBytes := make([]byte, numDepth*12)
	for i, engine := range depthInputEngines {
		copy(depthBytes[i*12:], engine)
	}

	fixedBytes := make([]byte, numFixed*13)
	for i, engine := range fixedInputEngines {
		copy(fixedBytes[i*13:], engine)
	}

	scoreDiffsDepth := make([]C.int, numEngines*numDepth)
	scoreDiffsFixed := make([]C.int, numEngines*numFixed)

	if numDepth > 0 {
		C.simulateDepthVsDepthCUDA((*C.char)(unsafe.Pointer(&enginesBytes[0])), C.int(numEngines),
			(*C.char)(unsafe.Pointer(&depthBytes[0])), C.int(numDepth),
			(*C.int)(unsafe.Pointer(&scoreDiffsDepth[0])))
	}
	if numFixed > 0 {
		C.simulateDepthVsFixedCUDA((*C.char)(unsafe.Pointer(&enginesBytes[0])), C.int(numEngines),
			(*C.char)(unsafe.Pointer(&fixedBytes[0])), C.int(numFixed),
			(*C.int)(unsafe.Pointer(&scoreDiffsFixed[0])))
	}

	totalScores := make([]int, numEngines)
	for i := 0; i < numEngines; i++ {
		for j := 0; j < numDepth; j++ {
			totalScores[i] += int(scoreDiffsDepth[i*numDepth+j])
		}
		for j := 0; j < numFixed; j++ {
			totalScores[i] += int(scoreDiffsFixed[i*numFixed+j])
		}
	}
	return totalScores
}

func evaluateBatch(engines []string, inputEngines []string, top100000Chan chan<- engineResult, progressComparisons *int64, numInputEngines int) {
    const subBatchSize = 20000 // Aantal engines per subbatch, pas dit aan naar wens
    var depthInputEngines, fixedInputEngines []string
    for _, engine := range inputEngines {
        if len(engine) == 12 {
            depthInputEngines = append(depthInputEngines, engine)
        } else if len(engine) == 13 {
            fixedInputEngines = append(fixedInputEngines, engine)
        }
    }

    for i := 0; i < len(engines); i += subBatchSize {
        end := i + subBatchSize
        if end > len(engines) {
            end = len(engines)
        }
        subBatch := engines[i:end]

        // Voer de berekeningen uit voor deze subbatch
        totalScores := evaluateBatchGPU(subBatch, depthInputEngines, fixedInputEngines)

        // Werk de voortgang bij
        progressMutex.Lock()
        atomic.AddInt64(progressComparisons, int64(len(subBatch)) * int64(numInputEngines))
        progressMutex.Unlock()

        // Verwerk de resultaten (bijv. top 100.000 scores)
        h := &minHeap{}
        heap.Init(h)
        maxSize := 100000
        for j, engine := range subBatch {
            totalScore := totalScores[j]
            if totalScore != 0 {
                result := engineResult{engine: engine, score: totalScore}
                if h.Len() < maxSize {
                    heap.Push(h, result)
                } else if totalScore > (*h)[0].score {
                    heap.Pop(h)
                    heap.Push(h, result)
                }
            }
        }
        for h.Len() > 0 {
            top100000Chan <- heap.Pop(h).(engineResult)
        }
    }
}

// **Hoofdfunctie**
func parseEngineCode(input string) string {
	parts := strings.Split(input, ":")
	engine := strings.TrimSpace(input)
	if len(parts) > 2 {
		engine = strings.TrimSpace(parts[2])
	}
	if len(engine) > 12 && strings.ContainsAny(engine, "12345") && !strings.ContainsAny(engine, "WVALD") {
		return engine[:12]
	}
	return engine
}

func main() {
	for {
		var inputEngines []string
		fmt.Println("Voer engine codes in (één per regel, '.' om te stoppen):")
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			input := strings.TrimSpace(scanner.Text())
			if input == "." || input == "" {
				break
			}
			engine := parseEngineCode(input)
			validDepth := len(engine) == 12 && !strings.ContainsAny(engine, "67890") && strings.ContainsAny(engine, "12345")
			validFixed := len(engine) == 13 && strings.ContainsAny(engine, "WVALD") && !strings.ContainsAny(engine, "1234567890")
			if validDepth || validFixed {
				inputEngines = append(inputEngines, engine)
			} else {
				fmt.Printf("Ongeldige engine code '%s'.\n", engine)
			}
		}
		if len(inputEngines) == 0 {
			fmt.Println("Geen engine codes ingevoerd. Gestopt.")
			break
		}

		var startDepth string
		var maxMemoryMB int
		fmt.Println("Voer de startdepth in (leeg voor alle combinaties, bijv. '51'): ")
		fmt.Scanln(&startDepth)
		if len(startDepth) > 12 || (startDepth != "" && strings.ContainsAny(startDepth, "67890")) {
			fmt.Println("Ongeldige startdepth.")
			continue
		}

		fmt.Println("Voer het maximale geheugen in MB in (default 128000): ")
		var memoryInput string
		fmt.Scanln(&memoryInput)
		if memoryInput == "" {
			maxMemoryMB = 128000
		} else if n, err := fmt.Sscanf(memoryInput, "%d", &maxMemoryMB); err != nil || n != 1 || maxMemoryMB < 1 {
			maxMemoryMB = 128000
			fmt.Println("Ongeldige invoer, defaulting naar 128,000 MB.")
		}

		generatedEngines := generateEngines(startDepth)
		const bytesPerResult = 24
		maxBufferSize := (maxMemoryMB * 1024 * 1024) / bytesPerResult
		if maxBufferSize > len(generatedEngines) {
			maxBufferSize = len(generatedEngines)
		}
		if maxBufferSize < 100000 {
			maxBufferSize = 100000
		}

		totalEngines := len(generatedEngines)
		numInputEngines := len(inputEngines)
		totalMatches := int64(totalEngines) * int64(numInputEngines)

		top100000Chan := make(chan engineResult, 10000000)
		var wg sync.WaitGroup
		var progressComparisons int64
		doneChan := make(chan struct{})

		// Start de voortgangsrapportage goroutine
		go func(totalMatches int64) {
			ticker := time.NewTicker(5 * time.Second)
			defer ticker.Stop()
			startTime := time.Now()
			for {
				select {
				case <-ticker.C:
					progressMutex.Lock()
					p := atomic.LoadInt64(&progressComparisons)
					progressMutex.Unlock()
					if p > 0 && p <= totalMatches {
						elapsed := time.Since(startTime).Seconds()
						speed := float64(p) / elapsed / 1000
						fmt.Printf("Progress: %d / %d matches (%.2f%%), Speed: %.1f k matches/s\n",
							p, totalMatches, float64(p)/float64(totalMatches)*100, speed)
					}
				case <-doneChan:
					return
				}
			}
		}(totalMatches)

		numThreads := runtime.NumCPU()
		enginesPerThread := (totalEngines + numThreads - 1) / numThreads
		for i := 0; i < numThreads; i++ {
			start := i * enginesPerThread
			end := start + enginesPerThread
			if end > totalEngines {
				end = totalEngines
			}
			wg.Add(1)
			go func(threadStart, threadEnd int) {
				defer wg.Done()
				batch := generatedEngines[threadStart:threadEnd]
				evaluateBatch(batch, inputEngines, top100000Chan, &progressComparisons, numInputEngines)
			}(start, end)
		}

		go func() {
			wg.Wait()
			close(top100000Chan)
			close(doneChan) // Signaleer de voortgangsrapportage om te stoppen
		}()

		file, err := os.Create("top_100000_engines.txt")
		if err != nil {
			fmt.Printf("Fout bij het openen van bestand: %v\n", err)
			return
		}
		defer file.Close()

		top100000 := &minHeap{}
		heap.Init(top100000)
		maxSize := 100000
		for result := range top100000Chan {
			if top100000.Len() < maxSize {
				heap.Push(top100000, result)
			} else if result.score > (*top100000)[0].score {
				heap.Pop(top100000)
				heap.Push(top100000, result)
			}
		}

		if top100000.Len() > 0 {
			results := make([]engineResult, 0, maxSize)
			for top100000.Len() > 0 {
				results = append(results, heap.Pop(top100000).(engineResult))
			}
			for i := len(results) - 1; i >= 0; i-- {
				result := results[i]
				_, err := file.WriteString(fmt.Sprintf("%s (score: %d)\n", result.engine, result.score))
				if err != nil {
					fmt.Printf("Fout bij het schrijven: %v\n", err)
					break
				}
			}
			fmt.Printf("Top 100,000 engines opgeslagen uit %d matches.\n", totalMatches)
		} else {
			fmt.Println("Geen engines geëvalueerd.")
		}
	}
	fmt.Println("Gestopt.")
}
