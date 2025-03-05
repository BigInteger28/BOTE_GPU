*COMPILE  
nvcc -shared -o simulate.dll simulate.cu -DSIMULATE_EXPORTS -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart  
go build main.go
