import cv2
import numpy as np
import pytesseract
import pandas as pd
import streamlit as st

from sudoku import solve_sudoku

# Previously defined functions (create_sudoku_solver, solve_sudoku) go here

def process_image(uploaded_image):
    # Convert the image to a grayscale image
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Calculate kernel size dynamically based on image dimensions
    height, width = gray.shape[:2]
    kernel_size = max(3, (min(height, width) // 50) | 1)  # Ensure it's odd

    # Use adaptive thresholding to binarize
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, kernel_size, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a rectangle
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Ensure we have four corners
    if len(approx_corners) != 4:
        raise ValueError("Could not find a valid Sudoku grid.")

    # Rectify the image using the four corners
    corners = np.array([corner[0] for corner in approx_corners], dtype='float32')

    # Determine the order of the corners: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]])

    ordered_corners = order_points(corners)

    # Establish the width and height for the new image
    (tl, tr, br, bl) = ordered_corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Rectify the image to a bird's-eye view of the grid
    destination = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_corners, destination)
    rectified = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
    
    # Apply binary threshold to the rectified image
    _, binary = cv2.threshold(rectified, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create image for visual output
    visual = cv2.cvtColor(rectified, cv2.COLOR_GRAY2RGB)

    # Calculate grid size for splitting
    grid_size_x = binary.shape[1] // 9
    grid_size_y = binary.shape[0] // 9

    # Divide the rectified image into 9x9 grids and analyze each cell
    puzzle = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = np.copy(binary[i*grid_size_y:(i+1)*grid_size_y, j*grid_size_x:(j+1)*grid_size_x])
            # Perform OCR on the cell
            digit = pytesseract.image_to_string(cell, config='--psm 10 digits')
            try:
                digit_value = int(digit.strip()) if digit.strip().isdigit() else 0
                digit_value = digit_value % 10 # make it one digit
            except ValueError:
                digit_value = 0
            row.append(digit_value)
            # Draw detected digit
            if digit_value != 0:
                cv2.putText(visual, str(digit_value), (j*grid_size_x + 10, (i+1)*grid_size_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, width / 256, (255, 0, 0), width // 256 + 1)
        puzzle.append(row)

    return puzzle, visual

@st.cache_data
def sudoku_from_image(image_file):
    # Load the image
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

    return process_image(image) 

def main():
    st.title("Sudoku Solver with Camera Input")

    st.write("Upload an image of the Sudoku puzzle.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, "Sudoku", width=256)
        # Process the uploaded image and extract the Sudoku puzzle
        puzzle, visual = sudoku_from_image(uploaded_file)

        # Display the rectified grid with detected digits
        st.image(visual, caption="Rectified Sudoku Grid with Detected Digits", width=256)

        st.write("Detected Sudoku Puzzle:")
        puzzle = st.data_editor(pd.DataFrame(puzzle), num_rows="dynamic", on_change=None)
        puzzle = puzzle.astype(int).values.tolist()

        if st.button("Solve Sudoku"):
            # Solve the Sudoku puzzle
            solution = solve_sudoku(puzzle)
            if solution:
                st.write("Solved Sudoku Puzzle:")
                st.dataframe(pd.DataFrame(solution))
            else:
                st.error("No solution found!")

if __name__ == "__main__":
    main()