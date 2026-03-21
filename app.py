# Library imports
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from numba import cuda
import time

from kernels import convolute_2d, convolute_3d

# Function to get the edge detection filter based on size
def generate_sobel_filter(ksize, direction="x"):
    # Get the 1D vectors from OpenCV
    if direction == 'x':
        # dx=1, dy=0 for horizontal edges
        kx, ky = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize)
    else:
        # dx=0, dy=1 for vertical edges
        kx, ky = cv2.getDerivKernels(dx=0, dy=1, ksize=ksize)
    
    # Multiply the 1D vectors to create the 2D matrix
    sobel_matrix = np.outer(ky, kx)
    
    # Return as float32
    return sobel_matrix.astype(np.float32)

# Hidden Warm-up Run for 2d kernel(Forces Numba to compile before the timer starts)
dummy_img = cuda.to_device(np.zeros((10, 10), dtype=np.float32))
dummy_filter = cuda.to_device(np.zeros((3, 3), dtype=np.float32))
dummy_out = cuda.device_array_like(dummy_img)
convolute_2d[(1, 1), (16, 16)](dummy_img, dummy_filter, dummy_out, 1)
cuda.synchronize()

# Hidden Warm-up Run for 3d kernel(Forces Numba to compile before the timer starts)
dummy_img = cuda.to_device(np.zeros((10, 10, 3), dtype=np.float32))
dummy_filter = cuda.to_device(np.zeros((3, 3), dtype=np.float32))
dummy_out = cuda.device_array_like(dummy_img)
convolute_3d[(1, 1), (16, 16)](dummy_img, dummy_filter, dummy_out, 1)
cuda.synchronize()


# File uploaded for handling the image file
uploaded_file = st.file_uploader("Upload an image.", type = ["jpg", "jpeg", "png"])


# If image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption = "Uploaded image")

    # Define number of threads across x and y
    threads = 16

    # Convert image to numpy array
    image = plt.imread(uploaded_file).astype(np.float32)

    # Extract height and width
    height = image.shape[0]
    width = image.shape[1]
    
    # Calculate total pixels and convert to Megapixels for readability
    total_pixels = width * height
    megapixels = total_pixels / 1_000_000
    
    # Display the stats to the user
    st.caption(f"**Resolution:** {width} x {height} ({megapixels:.2f} Megapixels)")
    
    # For pngs, if alpha channel exists, drop it
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Calculate number of blocks across x and y
    blocks_x = (image.shape[1] + threads - 1) // threads
    blocks_y = (image.shape[0] + threads - 1) // threads

    # Transfer image to gpu
    gpu_image = cuda.to_device(image)

    # Options for type of filter
    filter_options = ["Blur", "Edge Detection"]

    # Filter choice
    filter_type = st.radio("Choose which filter is to be applied:", options = filter_options)
    
    # Choose filter size
    filter_size = st.slider("Choose filter size:", min_value = 3, max_value = 11, step = 2, value = None)

    # Initialize time variables
    gpu_time = 0.0
    cpu_time = 0.0

    # Initialize image variables
    cpu_output_image = None
    gpu_output_image = None
    
    # Check if both values are provided
    if filter_type is not None and filter_size is not None:
        # Added a button for separating computation and submission
        if st.button("Apply filter"):
            # Added spinner for visual effect
            with st.spinner("Crunching pixels on the GPU..."):
                # If filter is blur
                if filter_type == "Blur":
                    # Temporary image variable
                    ans_image = None

                    # Define the denominator for the filter value
                    denom = filter_size ** 2

                    # Define filter array
                    filter = np.full(shape = (filter_size, filter_size), fill_value = 1 / denom, dtype = np.float32)
                    
                    # Transfer filter array to gpu 
                    gpu_filter = cuda.to_device(filter)

                    # Output array
                    out_image = cuda.device_array_like(gpu_image)

                    # Calculate radius
                    radius = filter_size // 2

                    # Record start time for GPU
                    gpu_start_time = time.perf_counter()

                    # Grayscale image
                    if image.ndim == 2:
                        # Launch the 2d kernel
                        convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter, out_image, radius)

                        # Synchronize
                        cuda.synchronize()
                    
                    # RGB image
                    elif image.ndim == 3:
                        # Edge case where grayscale has an extra dimension
                        if image.shape[2] == 1:
                            # Remove last dimension
                            gpu_image = np.squeeze(gpu_image, axis = -1)

                            # Launch the 2d kernel
                            convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter, out_image, radius)

                            # Synchronize
                            cuda.synchronize()
                        
                        elif image.shape[2] > 2:
                            # Launch the 3d kernel
                            convolute_3d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter, out_image, radius)

                            # Synchronize
                            cuda.synchronize()
                        
                    # Copy output array to cpu
                    ans_image = out_image.copy_to_host()

                    # Record end time for GPU
                    gpu_end_time = time.perf_counter()

                    # Calculcate GPU time
                    gpu_time = gpu_end_time - gpu_start_time

                    # Remove outliers and convert to uint8 for display
                    ans_clipped = np.clip(ans_image, 0, 255)
                    gpu_output_image = ans_clipped.astype(np.uint8)

                    # Record start time for CPU
                    cpu_start_time = time.perf_counter()

                    # Apply general blur filter on the image
                    cpu_output_image = cv2.filter2D(image, -1, filter)

                    # Record end time for CPU
                    cpu_end_time = time.perf_counter()

                    # Remove outliers and convert to uint8 for display
                    cpu_output_image = np.clip(cpu_output_image, 0, 255).astype(np.uint8)

                    # Calculate CPU time
                    cpu_time = cpu_end_time - cpu_start_time

                # If filter is edge detection
                else:
                    # Temporary image variable
                    ans_image = None

                    # Get both filters for x and y directions
                    filter_x = generate_sobel_filter(filter_size, "x")
                    filter_y = generate_sobel_filter(filter_size, "y") 

                    # Transfer filter arrays to gpu 
                    gpu_filter_x = cuda.to_device(filter_x)
                    gpu_filter_y = cuda.to_device(filter_y)

                    # Two output arrays for both directions
                    out_image_x = cuda.device_array_like(gpu_image)
                    out_image_y = cuda.device_array_like(gpu_image)

                    # Calculate radius
                    radius = filter_size // 2

                    # Record start time for GPU
                    gpu_start_time = time.perf_counter()

                    # Grayscale image
                    if image.ndim == 2:
                        # Launch the 2d kernel for x
                        convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_x, out_image_x, radius)

                        # Launch the 2d kernel for y
                        convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_y, out_image_y, radius)

                        # Synchronize
                        cuda.synchronize()
                    
                    # RGB image
                    elif image.ndim == 3:
                        # Edge case where grayscale has an extra dimension
                        if image.shape[2] == 1:
                            # Remove last dimension
                            gpu_image = np.squeeze(gpu_image, axis = -1)

                            # Launch the 2d kernel for x
                            convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_x, out_image_x, radius)

                            # Launch the 2d kernel for y
                            convolute_2d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_y, out_image_y, radius)

                            # Synchronize
                            cuda.synchronize()
                        
                        elif image.shape[2] > 2:
                            # Launch the 3d kernel for x
                            convolute_3d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_x, out_image_x, radius)

                             # Launch the 3d kernel for y
                            convolute_3d[(blocks_x, blocks_y), (threads, threads)](gpu_image, gpu_filter_y, out_image_y, radius)

                            # Synchronize
                            cuda.synchronize()
                    
                    # Copy the output back to cpu
                    edges_x = out_image_x.copy_to_host()
                    edges_y = out_image_y.copy_to_host()

                    # Record end time for GPU
                    gpu_end_time = time.perf_counter()

                    # Calculcate GPU time
                    gpu_time = gpu_end_time - gpu_start_time

                    # Combine both results for the final image
                    ans_image = np.sqrt(edges_x ** 2 + edges_y ** 2)

                    # Normalize and convert to uint8 for display
                    gpu_output_image = cv2.normalize(ans_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # Record start time for CPU
                    cpu_start_time = time.perf_counter()

                    # Apply Sobel operator
                    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = filter_size)  # Horizontal edges
                    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = filter_size)  # Vertical edges
                    
                    # Compute gradient magnitude
                    gradient_magnitude = cv2.magnitude(sobelx, sobely)
                    
                    # Normalize and convert to uint8 for display
                    cpu_output_image = cv2.normalize(gradient_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # Record end time for CPU
                    cpu_end_time = time.perf_counter()

                    # Calculate CPU time
                    cpu_time = cpu_end_time - cpu_start_time
    
            # Calculate the speedup
            speedup = cpu_time / gpu_time
            
            # Initialize columns in streamlit
            col1, col2 = st.columns(2)

            # Column 1: CPU
            with col1:
                # Header
                st.header("CPU output")
                
                # Display the image
                st.image(cpu_output_image)

                # Write the time
                st.subheader(f"Compute time: {cpu_time}")
            
            # Column 2: GPU
            with col2:
                # Header
                st.header("GPU output")
                
                # Display the image
                st.image(gpu_output_image)

                # Write the time
                st.subheader(f"Compute time: {gpu_time}")

            st.metric("GPU speed up:", value = speedup, delta = (cpu_time - gpu_time))
