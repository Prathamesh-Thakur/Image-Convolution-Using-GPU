# Image Convolution Using GPU

A high-performance GPU-accelerated image processing application that demonstrates the power of parallel computing for image convolution operations. This project compares GPU and CPU implementations of image filtering techniques, showcasing significant performance improvements through CUDA-based computation.

## Table of Contents
- [Project Overview](#project-overview)
- [Demo Video](#demo-video)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Performance Comparison](#performance-comparison)
- [Supported Filters](#supported-filters)
- [Project Structure](#project-structure)

## Project Overview

This project investigates how GPU acceleration can dramatically improve image convolution performance compared to traditional CPU-based approaches. It provides an interactive web interface built with Streamlit where users can upload images and apply various filters in real-time, with live performance metrics comparing GPU and CPU execution times.

### Motivation

Image convolution is a fundamental operation in computer vision and digital signal processing. Traditional CPU-based implementations process pixels sequentially, which can be slow for large images. By leveraging GPU parallelism, we can process thousands of pixels simultaneously, achieving significant speedups. This project demonstrates these principles practically with a modern, user-friendly interface.

## Demo Video

📹 **Watch the project in action**: [Link](https://drive.google.com/file/d/12PdgFTfPtRcDMOd-_psw39OYJ7Q03Tcl/view?usp=sharing)

## Features

### Core Features
- **GPU-Accelerated Convolution**: Custom CUDA kernels for fast image processing
- **Dual-Implementation Support**: Built-in CPU comparison for performance benchmarking
- **Multiple Filter Types**:
  - Blur filters (customizable kernel sizes from 3×3 to 11×11)
  - Sobel edge detection (directional edge detection in X and Y directions)
- **Image Format Support**: 
  - Grayscale images (2D convolutional kernels)
  - RGB color images (3D convolutional kernels with per-channel processing)
  - PNG, JPG, and JPEG formats
- **Interactive Web Interface**: Built with Streamlit for ease of use
- **Performance Metrics**: Real-time display of GPU vs CPU computation times and speedup ratios
- **Optimized Memory Management**: Uses shared memory for efficient data reuse on GPU

### Technical Optimizations
- **Shared Memory Utilization**: Reduces global memory bandwidth requirements
- **Halo Padding**: Efficient handling of filter boundary conditions
- **Block-Level Synchronization**: Proper thread coordination for data consistency
- **Automatic CUDA Compilation**: Numba JIT compilation with warm-up runs to eliminate compilation overhead

## Technical Architecture

### CUDA Kernel Design

The project implements two specialized CUDA kernels for different image dimensions:

#### 2D Kernel (`convolute_2d`)
- **Purpose**: Handles grayscale images and single-channel processing
- **Block Size**: 16×16 threads per block
- **Shared Memory**: Dynamically allocated based on filter radius (up to 26×26 for 11×11 filters)
- **Boundary Handling**: Zero-padding for out-of-bounds pixels

#### 3D Kernel (`convolute_3d`)
- **Purpose**: Processes RGB images by applying the same filter to each color channel independently
- **Block Size**: 16×16 threads per block (thread grid covers 2D spatial dimensions)
- **Channel Processing**: Each thread handles all 3 color channels for its pixel
- **Shared Memory**: 3D shared memory structure (26×26×3) for RGB channel data

### Grid and Block Organization
- **Thread Blocks**: 16×16 threads per block for optimal occupancy
- **Grid Dimensions**: Automatically calculated to cover the entire image
  - `blocks_x = (width + 16 - 1) // 16`
  - `blocks_y = (height + 16 - 1) // 16`
- **Shared Memory Strategy**: Load entire halo (filter padding) into shared memory before computation to minimize global memory accesses

### Memory Access Pattern
```
Global Memory → Shared Memory → Thread Registers → Computation
```

This hierarchical memory usage ensures:
1. Coalesced memory access from global memory to shared memory
2. Fast register-based computation within threads
3. Minimal repeated global memory reads

## Dependencies

### Core Libraries
- **NumPy**: Numerical array operations
- **Numba**: CUDA JIT compilation for GPU kernels
- **Streamlit**: Web application framework
- **OpenCV (cv2)**: Image I/O and CPU reference implementations
- **Matplotlib**: Image loading utilities
- **CUDA**: NVIDIA GPU compute platform (required for hardware acceleration)

### Hardware Requirements
- NVIDIA GPU with CUDA compute capability 3.0 or higher
- Latest NVIDIA CUDA Toolkit installed
- Compatible cuDNN library (recommended)

## Installation

### Step 1: Set Up Python Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install numpy numba streamlit opencv-python matplotlib
```

### Step 3: Verify CUDA Support
```python
from numba import cuda
print(cuda.is_available())  # Should print True if CUDA is properly configured
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Alternative: Using the Jupyter Notebook in Hosted Environments

For cloud-based environments such as **Google Colab**, **Kaggle Notebooks**, or **Jupyter Hub**, use the included notebook (`notebook.ipynb`):

#### Prerequisites
- Access to a hosted Jupyter environment with GPU support
- Your environment has internet access
- NVIDIA GPU available in the environment

#### Step-by-Step Setup in Google Colab

1. **Open the Notebook**
   - Upload `notebook.ipynb` to your Google Colab environment, or create a new notebook

2. **Upload Project Files**
   - Click the Files icon in the left sidebar
   - Upload `app.py` and `kernels.py` to the notebook's working directory
   - These files are required for the application to run

3. **Run Cell 1: Install Streamlit**
   ```python
   !pip install -q streamlit
   ```
   - This installs the Streamlit web framework
   - The `-q` flag runs the installation quietly

4. **Run Cell 2: Set Up Tunnel and Launch Application**
   ```python
   # 1. Install Cloudflare's tunnel tool
   !wget -q -O cloudflared-linux-amd64.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared-linux-amd64.deb
   
   # 2. Run Streamlit in the background
   !streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false &>/content/logs.txt &
   
   # 3. Create the Cloudflare Tunnel
   !cloudflared tunnel --url http://localhost:8501
   ```
   - Installs Cloudflared (exposes local services to the internet)
   - Launches Streamlit application on port 8501
   - Creates a public tunnel URL that you can access from your browser

5. **Access the Application**
   - The second cell will output a public URL (e.g., `https://xxx.trycloudflare.com`)
   - Copy and paste this URL into your browser
   - The GPU-accelerated image convolution app is now accessible!

#### Why Use a Notebook in Hosted Environments?

- ✅ **GPU Access**: Cloud environments provide free GPU access
- ✅ **No Installation**: All dependencies installed via pip
- ✅ **Public Sharing**: Cloudflare tunnel creates a shareable link
- ✅ **Ease of Use**: No need to manage local CUDA installations
- ✅ **Collaboration**: Share the tunnel URL with others for live demos

#### File Requirements for Notebook

When using the notebook approach, ensure these files are in the same directory:
- `notebook.ipynb` - The main notebook
- `app.py` - The Streamlit application
- `kernels.py` - The CUDA kernel implementations

#### Troubleshooting Notebook Setup

**"Streamlit command not found"**
- Re-run the `pip install streamlit` cell
- Wait for installation to complete before running the next cell

**"cloudflared not found"**
- The second cell installs cloudflared
- Make sure Debian packages can be installed (Linux/Colab environment)

**"Unable to connect to tunnel URL"**
- Wait 10-15 seconds for the tunnel to establish
- Check that the public URL appears in the cell output
- Ensure port 8501 is accessible

**CUDA Errors**
- Verify GPU is enabled in your Colab environment (Runtime → Change Runtime Type → GPU)
- Some free-tier GPUs may be under heavy load; try again later

## Usage

### Basic Workflow

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image**
   - Click "Upload an image" button
   - Select a PNG, JPG, or JPEG file
   - Image will be displayed with resolution information in megapixels

3. **Choose Filter Type**
   - **Blur**: Applies a uniform averaging filter
   - **Edge Detection**: Applies Sobel edge detection

4. **Select Filter Size**
   - Use the slider to choose kernel size (3, 5, 7, 9, or 11)
   - Larger kernels provide stronger effects but require more computation

5. **Apply Filter**
   - Click "Apply filter" button
   - The application processes the image on GPU while simultaneously processing on CPU
   - Results display side-by-side with computation times

6. **View Results**
   - Compare GPU and CPU outputs visually
   - Check computation times (in seconds)
   - Review speedup ratio: `GPU Speed up = CPU Time / GPU Time`

### Example Usage Scenarios

**Scenario 1: Quick Blur Effect**
- Upload a photo
- Select "Blur" filter
- Choose size 5
- Click apply and see instant results with speedup metrics

**Scenario 2: Edge Detection Analysis**
- Upload a cell or medical image
- Select "Edge Detection" filter
- Try different kernel sizes (3, 5, 7) to find optimal edge emphasis
- Compare GPU edge detection with OpenCV's Sobel implementation

**Scenario 3: Performance Benchmarking**
- Upload a high-resolution image (4K, 8K)
- Apply various filters and sizes
- Record speedup values for different configurations
- Analyze how image resolution affects GPU advantage

## Implementation Details

### Application Flow (`app.py`)

```
1. Import Libraries & Initialize
   ├─ Load CUDA kernels from kernels.py
   ├─ Warm-up runs for 2D and 3D kernels
   └─ Initialize Streamlit UI elements

2. Image Upload & Processing
   ├─ File uploader widget
   ├─ Read image with matplotlib
   ├─ Handle PNG alpha channels
   └─ Collect image statistics (resolution, megapixels)

3. Filter Configuration
   ├─ Filter type selection (Blur/Edge Detection)
   ├─ Kernel size slider
   └─ Block/thread grid calculation

4. Computation Pipeline
   ├─ Create filter kernels
   ├─ Transfer data to GPU
   ├─ Execute GPU kernel with timing
   ├─ Execute CPU reference with timing
   ├─ Calculate speedup ratio
   └─ Display results side-by-side
```

### GPU Kernel Execution (`kernels.py`)

#### Memory Loading Phase
```cuda
while thread_id < total_shared_pixels:
    Load pixel_value from global memory
    Store in shared memory with halo padding
    thread_id += block_size
```

#### Synchronization
```cuda
__syncthreads()  // Wait for all threads to finish loading
```

#### Convolution Phase
```cuda
for row in [-radius, radius]:
    for col in [-radius, radius]:
        accumulate += shared_mem[...] * filter[...]
output[thread_pos] = accumulated_value
```

### Filter Generation

**Blur Filter**
- Uniform weights: `1 / (size²)`
- Applied independently to each channel
- Example for 3×3: Each weight = 1/9

**Sobel Edge Detection**
- Two directional filters (X and Y gradients)
- Generated using OpenCV's `getDerivKernels()` function
- Final edge magnitude: `√(Gx² + Gy²)`
- Normalized to 0-255 range for display

## Performance Comparison

### Performance Factors

**GPU Advantages**
- ✅ Massive parallelism (thousands of threads)
- ✅ Optimized global memory bandwidth
- ✅ Shared memory for data reuse
- ✅ Scales with image size

**CPU Limitations**
- ❌ Sequential pixel processing
- ❌ Limited cache efficiency
- ❌ Single-threaded computation (standard OpenCV)

### Expected Speedup Ranges

| Image Size | Filter Size | Typical Speedup |
|-----------|------------|-----------------|
| 1 MP | 3×3 | 2-5× |
| 5 MP | 5×5 | 5-15× |
| 12 MP | 7×7 | 10-25× |
| 20+ MP | 11×11 | 15-30×+ |

**Note**: Actual speedup depends on GPU model, image format (grayscale vs RGB), and system load.

### Timing Overhead

The following overheads are included in GPU time measurements:
- Memory transfer (host → device)
- Kernel compilation (eliminated via warm-up runs)
- Kernel execution
- Memory transfer (device → host)
- Device synchronization

## Supported Filters

### 1. Blur (Box Filter)
**Formula**: 
$$I'(x,y) = \frac{1}{K^2} \sum_{i,j} I(x+i-c, y+j-c)$$
where $K$ is filter size and $c$ is center offset.

**Use Cases**:
- Image smoothing
- Noise reduction
- Pre-processing for other algorithms

**Kernel Sizes**: 3×3, 5×5, 7×7, 9×9, 11×11

### 2. Sobel Edge Detection
**X-Direction**: Emphasis on vertical edges
**Y-Direction**: Emphasis on horizontal edges

**Combined Magnitude**:
$$G = \sqrt{G_x^2 + G_y^2}$$

**Use Cases**:
- Edge detection
- Boundary identification
- Image segmentation
- Object detection preprocessing

**Kernel Sizes**: 3×3, 5×5, 7×7, 9×9, 11×11

## Project Structure

```
Image-Convolution-Using-GPU/
├── README.md                 # Project documentation
├── app.py                    # Main Streamlit application
├── kernels.py               # CUDA kernel implementations
├── requirements.txt         # Python dependencies
└── LICENSE                  # Project license
```

### File Descriptions

**app.py**
- Main application entry point
- Streamlit UI components
- Filter generation (Sobel filters via OpenCV)
- GPU/CPU execution orchestration
- Performance timing and display

**kernels.py**
- `convolute_2d`: CUDA kernel for 2D convolution (grayscale)
- `convolute_3d`: CUDA kernel for 3D convolution (RGB)
- Shared memory management
- Thread synchronization logic
- Boundary handling

**requirements.txt**
- All Python package dependencies
- Version specifications for compatibility

**LICENSE**
- Project licensing information

## Advanced Features

### Warm-up Runs
The application performs warm-up runs for both 2D and 3D kernels before timing:
```python
# Eliminates JIT compilation overhead from benchmark times
dummy_out = cuda.device_array_like(dummy_img)
convolute_2d[(1, 1), (16, 16)](dummy_img, dummy_filter, dummy_out, 1)
cuda.synchronize()
```

### Grayscale Handling
Automatically detects and handles multiple grayscale image formats:
- 2D arrays: Direct 2D kernel processing
- 3D arrays with 1 channel: Squeezed to 2D before processing
- PNG images: Alpha channel removed when present (4-channel → 3-channel)

### Dynamic Block Calculation
Grid dimensions automatically adjust to image size:
```python
blocks_x = (width + 16 - 1) // 16
blocks_y = (height + 16 - 1) // 16
blocks = (blocks_x, blocks_y)
threads = (16, 16)
```

## Troubleshooting

### Common Issues

**"CUDA not found"**
- Ensure NVIDIA GPU drivers are installed
- Verify CUDA Toolkit installation
- Check `nvidia-smi` command works

**"Filter size must be odd"**
- Slider only provides odd numbers (3, 5, 7, 9, 11)
- This is intentional for symmetric filter kernels

**"Memory error on large images"**
- GPU memory is limited
- Reduce image size or filter size
- Check available GPU memory with `nvidia-smi`

**"No speedup or slower on GPU"**
- Small images may have overhead exceeding benefits
- GPU shines with images > 1 megapixel
- Ensure GPU drivers are up-to-date

## Future Enhancements

Potential improvements and extensions:
- Additional filter types (Gaussian, Laplacian, Canny)
- Batch processing for multiple images
- Real-time video stream processing
- Custom filter kernel upload
- Advanced filter parameters (normalization options)
- Performance profiling dashboard
- Quantitative image quality metrics (PSNR, SSIM)
- GPU memory optimization for extreme resolutions
- Multi-GPU support for distributed processing

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

See LICENSE file for details.

## Author Notes

This project demonstrates the practical application of GPU computing to real-world image processing tasks. The ~10-30× speedup observed on modern GPUs showcases why GPU acceleration is essential for computationally intensive applications in computer vision, scientific computing, and machine learning.
