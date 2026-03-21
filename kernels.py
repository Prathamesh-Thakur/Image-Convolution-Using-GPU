import numpy as np
from numba import cuda, types

# Predefined constant values. Radius corresponds to a max filter size of 11x11.
thread_dim = 16
max_radius = 5
max_shared_dim = thread_dim + (2 * max_radius)

@cuda.jit
def convolute_2d(image_array, filter, output_image_array, radius):
  # Create shared memory for the block. Allocate maximum right now, and use subset
  shared_mem = cuda.shared.array((max_shared_dim, max_shared_dim), dtype = types.float32)

  # Define global thread indices
  x, y = cuda.grid(2)

  # Index when flattened
  local_id = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x

  # Total threads in block
  total_threads = cuda.blockDim.x * cuda.blockDim.y

  # Calculate required shared width and height based on given radius
  shared_width = thread_dim + (2 * radius)
  shared_height = thread_dim + (2 * radius)

  # Total pixels to be loaded in shared memory
  total_shared_pixels = shared_width * shared_height

  # Loop to load pixel values into shared memory
  i = local_id

  # Assign image pixel values to the shared memory
  while i < total_shared_pixels:
    # Calculate the x and y indices in the shared memory grid
    x_shared = i % shared_width
    y_shared = i // shared_width

    # Find the true start of the shared block by removing the padding
    halo_start_x = (cuda.blockIdx.x * cuda.blockDim.x) - radius
    halo_start_y = (cuda.blockIdx.y * cuda.blockDim.y) - radius

    # Add the shared x and y indices to find the pixel in the image array to be assigned
    target_global_x = halo_start_x + x_shared
    target_global_y = halo_start_y + y_shared

    # Read from global to shared (with boundary checks!)
    if (target_global_x >= 0 and target_global_x < image_array.shape[1] and target_global_y >= 0 and target_global_y < image_array.shape[0]):
        shared_mem[y_shared, x_shared] = image_array[target_global_y, target_global_x]
    else:
        shared_mem[y_shared, x_shared] = 0.0  # Zero padding for the edges

    # Add the total threads to enable jumping to the relevant pixels
    i += total_threads

  # Synchronize all the thread data
  cuda.syncthreads()

  # Get the coordinates for the center of the filter
  center_x = cuda.threadIdx.x + radius
  center_y = cuda.threadIdx.y + radius

  # Updated value of the pixel after applying the filter
  new_pixel_value = 0.0

  # Iterate over the rows
  for row in range(-radius, radius + 1):
    # Iterate over the columns
    for col in range(-radius, radius + 1):
      # Multiply the filter value by the old pixel value
      new_pixel_value += shared_mem[center_y + row, center_x + col] * filter[row + radius, col + radius]

  # Check if the thread coorindates are within the image
  if x < image_array.shape[1] and y < image_array.shape[0]:
    # Assign the pixel value to the image
    output_image_array[y, x] = new_pixel_value


@cuda.jit
def convolute_3d(image_array, filter, output_image_array, radius):
  # Create shared memory for the block. Allocate maximum right now, and use subset
  shared_mem = cuda.shared.array((max_shared_dim, max_shared_dim, 3), dtype = types.float32)

  # Define global thread indices
  x, y = cuda.grid(2)

  # Index when flattened
  local_id = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x

  # Total threads in block
  total_threads = cuda.blockDim.x * cuda.blockDim.y

  # Calculate required shared width and height based on given radius
  shared_width = thread_dim + (2 * radius)
  shared_height = thread_dim + (2 * radius)

  # Total pixels to be loaded in shared memory
  total_shared_pixels = shared_width * shared_height

  # Loop to load pixel values into shared memory
  i = local_id

  # Assign image pixel values to the shared memory
  while i < total_shared_pixels:
    # Calculate the x and y indices in the shared memory grid
    x_shared = i % shared_width
    y_shared = i // shared_width

    # Find the true start of the shared block by removing the padding
    halo_start_x = (cuda.blockIdx.x * cuda.blockDim.x) - radius
    halo_start_y = (cuda.blockIdx.y * cuda.blockDim.y) - radius

    # Add the shared x and y indices to find the pixel in the image array to be assigned
    target_global_x = halo_start_x + x_shared
    target_global_y = halo_start_y + y_shared

    # Read from global to shared (with boundary checks!)
    if (target_global_x >= 0 and target_global_x < image_array.shape[1] and target_global_y >= 0 and target_global_y < image_array.shape[0]):
        for c in range(3):
          shared_mem[y_shared, x_shared, c] = image_array[target_global_y, target_global_x, c]
    else:
        for c in range(3):
          shared_mem[y_shared, x_shared, c] = 0.0  # Zero padding for the edges

    # Add the total threads to enable jumping to the relevant pixels
    i += total_threads

  # Synchronize all the thread data
  cuda.syncthreads()

  # Get the coordinates for the center of the filter
  center_x = cuda.threadIdx.x + radius
  center_y = cuda.threadIdx.y + radius

  # Updated value of the pixel after applying the filter
  new_pixel_value_red = 0.0
  new_pixel_value_green = 0.0
  new_pixel_value_blue = 0.0

  # Iterate over the rows
  for row in range(-radius, radius + 1):
    # Iterate over the columns
    for col in range(-radius, radius + 1):
      # Multiply the filter value by the old pixel value
      new_pixel_value_red += shared_mem[center_y + row, center_x + col, 0] * filter[row + radius, col + radius]
      new_pixel_value_green += shared_mem[center_y + row, center_x + col, 1] * filter[row + radius, col + radius]
      new_pixel_value_blue += shared_mem[center_y + row, center_x + col, 2] * filter[row + radius, col + radius]

  # Check if the thread coorindates are within the image
  if x < image_array.shape[1] and y < image_array.shape[0]:
    # Assign the pixel value to the image
    output_image_array[y, x, 0] = new_pixel_value_red
    output_image_array[y, x, 1] = new_pixel_value_green
    output_image_array[y, x, 2] = new_pixel_value_blue