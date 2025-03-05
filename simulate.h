#ifndef SIMULATE_H
#define SIMULATE_H

#ifdef _WIN32
    #ifdef SIMULATE_EXPORTS
        #define SIMULATE_API __declspec(dllexport)
    #else
        #define SIMULATE_API __declspec(dllimport)
    #endif
#else
    #define SIMULATE_API
#endif

// Make the header C-compatible
#ifdef __cplusplus
extern "C" {
#endif

SIMULATE_API void simulateDepthVsDepthCUDA(const char* generatedEngines, int numGenerated,
                                           const char* depthInputEngines, int numDepth, int* scoreDiffs);
SIMULATE_API void simulateDepthVsFixedCUDA(const char* generatedEngines, int numGenerated,
                                           const char* fixedInputEngines, int numFixed, int* scoreDiffs);

#ifdef __cplusplus
}
#endif

#endif // SIMULATE_H