export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 CUDACXX=$(command -v nvcc)
export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
~/.local/share/aivoice/venv/bin/pip install --no-cache-dir -c ~/.local/share/aivoice/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 'llama-cpp-python>=0.3' 
