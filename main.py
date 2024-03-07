import numpy as np
import rich
from rich.console import Console

console = Console()


def is_positive_semidefinite(matrix):
  """
  This function checks if a matrix is positive semidefinite.

  Args:
      matrix: A numpy array representing the matrix.

  Returns:
      True if the matrix is positive semidefinite, False otherwise.
  """
  return np.all(np.linalg.eigvalsh(matrix) >= 0)


def generate_gaussian_matrix(n):
  """
  This function generates a random nxn matrix with Gaussian entries.

  Args:
      n: The size of the square matrix (nxn).

  Returns:
      A numpy array representing the random Gaussian matrix.
  """
  # Generate random matrix with standard normal distribution
  random_matrix = np.random.randn(n, n)

  # Ensure the matrix is symmetric (positive semidefinite matrices are symmetric)
  return random_matrix + random_matrix.T


# Example usage
n = 3  # Replace with your desired matrix size
matrix = generate_gaussian_matrix(n)

console.print(matrix)
console.print(is_positive_semidefinite(matrix))

# Create n by n identity matrix
identity_matrix = np.eye(n)
console.print(identity_matrix)
console.print(is_positive_semidefinite(identity_matrix))

# Add 1 to each entry in the matrix
matrix += identity_matrix * 3
console.print(matrix)
console.print(np.linalg.eigvalsh(matrix))
console.print(is_positive_semidefinite(matrix))
