#include <cuda_runtime.h>
#include "simulate.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// **Helperfuncties voor gebruik op de GPU**
__device__ int determineWinner(char move1, char move2) {
    int move1Idx = -1;
    if (move1 == 'W') move1Idx = 0;
    else if (move1 == 'V') move1Idx = 1;
    else if (move1 == 'A') move1Idx = 2;
    else if (move1 == 'L') move1Idx = 3;
    else if (move1 == 'D') move1Idx = 4;

    int move2Idx = -1;
    if (move2 == 'W') move2Idx = 0;
    else if (move2 == 'V') move2Idx = 1;
    else if (move2 == 'A') move2Idx = 2;
    else if (move2 == 'L') move2Idx = 3;
    else if (move2 == 'D') move2Idx = 4;

    if (move1Idx == -1 || move2Idx == -1) return 0;

    const int moveWins[5][5] = {
        {0, 1, 0, 2, 0}, // W
        {2, 0, 1, 0, 0}, // V
        {0, 2, 0, 1, 0}, // A
        {1, 0, 2, 0, 0}, // L
        {0, 0, 0, 0, 0}  // D
    };
    return moveWins[move1Idx][move2Idx];
}

__device__ int getIndex(char move) {
    switch (move) {
        case 'W': return 0;
        case 'V': return 1;
        case 'A': return 2;
        case 'L': return 3;
        case 'D': return 4;
        default: return -1;
    }
}

__device__ char getElementFromCode(int depth) {
    if (depth < 1 || depth > 5) return 0;
    const char depthToElement[5] = {'W', 'V', 'A', 'L', 'D'};
    return depthToElement[depth - 1];
}

__device__ char getElementByDepth(char prevElement, int depth) {
    if (depth == 5) return 'D';
    if (prevElement == 0) return 0;
    if (prevElement == 'D') prevElement = 'L';
    if (depth < 1 || depth > 4) return 0;
    const char elementsDepth[4][4] = {
        {'L', 'A', 'V', 'W'}, // W
        {'W', 'L', 'A', 'V'}, // V
        {'V', 'W', 'L', 'A'}, // A
        {'A', 'V', 'W', 'L'}  // L
    };
    int idx = -1;
    if (prevElement == 'W') idx = 0;
    else if (prevElement == 'V') idx = 1;
    else if (prevElement == 'A') idx = 2;
    else if (prevElement == 'L') idx = 3;
    if (idx == -1) return 0;
    return elementsDepth[idx][depth - 1];
}

__device__ char chooseAvailableElement(char target, int* available) {
    int targetIdx = getIndex(target);
    if (targetIdx != -1 && available[targetIdx] > 0) {
        return target;
    }
    char current = target;
    for (int i = 0; i < 5; i++) {
        if (current == 'W') current = 'L';
        else if (current == 'V') current = 'W';
        else if (current == 'A') current = 'V';
        else if (current == 'L') current = 'A';
        else break;
        int currentIdx = getIndex(current);
        if (currentIdx != -1 && available[currentIdx] > 0) {
            return current;
        }
    }
    if (available[4] > 0) { // D
        return 'D';
    }
    return 0;
}

__device__ char getLastElement(int* available) {
    const char candidates[5] = {'W', 'V', 'A', 'L', 'D'};
    for (int i = 0; i < 5; i++) {
        char c = candidates[i];
        int idx = getIndex(c);
        if (available[idx] > 0) {
            return c;
        }
    }
    return 0;
}

// **CUDA Kernels**
__global__ void simulateDepthVsFixedKernel(const char* generatedEngines, const char* fixedInputEngines,
                                           int numGenerated, int numFixed, int* scoreDiffs) {
    int generatedIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int fixedIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (generatedIdx < numGenerated && fixedIdx < numFixed) {
        const char* engine = generatedEngines + generatedIdx * 12;
        const char* fixedEngine = fixedInputEngines + fixedIdx * 13;
        
        int available[5] = {3, 3, 3, 3, 1}; // W, V, A, L, D
        char moves[13];
        
        for (int i = 0; i < 12; i++) {
            int depth = engine[i] - '0';
            char prevMove = (i == 0) ? 0 : fixedEngine[i-1];
            char target = (i == 0) ? getElementFromCode(depth) : getElementByDepth(prevMove, depth);
            char move = chooseAvailableElement(target, available);
            if (move == 0) {
                move = 'W'; // Default
            }
            int moveIdx = getIndex(move);
            available[moveIdx]--;
            moves[i] = move;
        }
        
        char lastMove = getLastElement(available);
        if (lastMove != 0) {
            int lastIdx = getIndex(lastMove);
            available[lastIdx]--;
            moves[12] = lastMove;
        } else {
            moves[12] = 'W';
        }
        
        int p1Score = 0, p2Score = 0;
        for (int i = 0; i < 13; i++) {
            char move1 = moves[i];
            char move2 = fixedEngine[i];
            int winner = determineWinner(move1, move2);
            if (winner == 1) p1Score++;
            else if (winner == 2) p2Score++;
        }
        
        // Nieuwe scoreberekening
        int diff = p1Score - p2Score;
        if (p1Score > p2Score) {
            scoreDiffs[generatedIdx * numFixed + fixedIdx] = diff + 10; // Winst: +10
        } else if (p1Score < p2Score) {
            scoreDiffs[generatedIdx * numFixed + fixedIdx] = diff - 10; // Verlies: -10
        } else {
            scoreDiffs[generatedIdx * numFixed + fixedIdx] = p1Score;   // Gelijkspel: p1Score
        }
    }
}

// Kernel voor Depth vs Depth
__global__ void simulateDepthVsDepthKernel(const char* generatedEngines, const char* depthInputEngines,
                                           int numGenerated, int numDepth, int* scoreDiffs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < numGenerated && idy < numDepth) {
        const char* engine1 = generatedEngines + idx * 12;
        const char* engine2 = depthInputEngines + idy * 12;
        
        int available1[5] = {3, 3, 3, 3, 1};
        int available2[5] = {3, 3, 3, 3, 1};
        char moves1[13], moves2[13];
        int p1Score = 0, p2Score = 0;
        
        for (int i = 0; i < 12; i++) {
            int depth1 = engine1[i] - '0';
            int depth2 = engine2[i] - '0';
            char prevMove2 = (i == 0) ? 0 : moves2[i-1];
            char prevMove1 = (i == 0) ? 0 : moves1[i-1];
            
            char target1 = (i == 0) ? getElementFromCode(depth1) : getElementByDepth(prevMove2, depth1);
            char target2 = (i == 0) ? getElementFromCode(depth2) : getElementByDepth(prevMove1, depth2);
            
            char move1 = chooseAvailableElement(target1, available1);
            if (move1 == 0) move1 = 'W';
            char move2 = chooseAvailableElement(target2, available2);
            if (move2 == 0) move2 = 'W';
            
            available1[getIndex(move1)]--;
            available2[getIndex(move2)]--;
            moves1[i] = move1;
            moves2[i] = move2;
            
            int winner = determineWinner(move1, move2);
            if (winner == 1) p1Score++;
            else if (winner == 2) p2Score++;
        }
        
        char lastMove1 = getLastElement(available1);
        if (lastMove1 != 0) available1[getIndex(lastMove1)]--;
        else lastMove1 = 'W';
        char lastMove2 = getLastElement(available2);
        if (lastMove2 != 0) available2[getIndex(lastMove2)]--;
        else lastMove2 = 'W';
        moves1[12] = lastMove1;
        moves2[12] = lastMove2;
        
        int winner = determineWinner(lastMove1, lastMove2);
        if (winner == 1) p1Score++;
        else if (winner == 2) p2Score++;
        
        // Nieuwe scoreberekening
        int diff = p1Score - p2Score;
        if (p1Score > p2Score) {
            scoreDiffs[idx * numDepth + idy] = diff + 10; // Winst: +10
        } else if (p1Score < p2Score) {
            scoreDiffs[idx * numDepth + idy] = diff - 10; // Verlies: -10
        } else {
            scoreDiffs[idx * numDepth + idy] = p1Score;   // Gelijkspel: p1Score
        }
    }
}

// **Wrapper-functies voor cgo**
extern "C" void simulateDepthVsFixedCUDA(const char* generatedEngines, int numGenerated,
                                         const char* fixedInputEngines, int numFixed, int* scoreDiffs) {
    char *d_generatedEngines, *d_fixedInputEngines;
    int *d_scoreDiffs;

    // Alloceer geheugen op de GPU met de juiste grootte
    cudaMalloc(&d_generatedEngines, numGenerated * 12 * sizeof(char));  // 12 bytes per generated engine
    cudaMalloc(&d_fixedInputEngines, numFixed * 13 * sizeof(char));     // 13 bytes per fixed engine
    cudaMalloc(&d_scoreDiffs, numGenerated * numFixed * sizeof(int));

    // Kopieer data van host naar GPU met de juiste grootte
    cudaMemcpy(d_generatedEngines, generatedEngines, numGenerated * 12 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fixedInputEngines, fixedInputEngines, numFixed * 13 * sizeof(char), cudaMemcpyHostToDevice);

    // Definieer block- en griddimensies
    dim3 blockDim(32, 32);
    dim3 gridDim((numGenerated + blockDim.x - 1) / blockDim.x, (numFixed + blockDim.y - 1) / blockDim.y);

    // Start de kernel
    simulateDepthVsFixedKernel<<<gridDim, blockDim>>>(d_generatedEngines, d_fixedInputEngines, numGenerated, numFixed, d_scoreDiffs);

    // Kopieer resultaten terug naar host
    cudaMemcpy(scoreDiffs, d_scoreDiffs, numGenerated * numFixed * sizeof(int), cudaMemcpyDeviceToHost);

    // Vrij GPU-geheugen
    cudaFree(d_generatedEngines);
    cudaFree(d_fixedInputEngines);
    cudaFree(d_scoreDiffs);
}

// Wrapper voor Depth vs Depth
extern "C" SIMULATE_API void simulateDepthVsDepthCUDA(const char* generatedEngines, int numGenerated,
                                                      const char* depthInputEngines, int numDepth, int* scoreDiffs) {
    char *d_generatedEngines, *d_depthInputEngines;
    int *d_scoreDiffs;
    cudaMalloc(&d_generatedEngines, numGenerated * 12 * sizeof(char));
    cudaMalloc(&d_depthInputEngines, numDepth * 12 * sizeof(char));
    cudaMalloc(&d_scoreDiffs, numGenerated * numDepth * sizeof(int));
    cudaMemcpy(d_generatedEngines, generatedEngines, numGenerated * 12 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depthInputEngines, depthInputEngines, numDepth * 12 * sizeof(char), cudaMemcpyHostToDevice);
    dim3 blockDim(32, 32);
    dim3 gridDim((numGenerated + blockDim.x - 1) / blockDim.x, (numDepth + blockDim.y - 1) / blockDim.y);
    simulateDepthVsDepthKernel<<<gridDim, blockDim>>>(d_generatedEngines, d_depthInputEngines, numGenerated, numDepth, d_scoreDiffs);
    cudaMemcpy(scoreDiffs, d_scoreDiffs, numGenerated * numDepth * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_generatedEngines);
    cudaFree(d_depthInputEngines);
    cudaFree(d_scoreDiffs);
}
