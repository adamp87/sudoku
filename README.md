
# Sudoku Solver with Binary Integer Programming

This is a small tool purely developed using ChatGPT 4o. The copy of the complete conversation is present in this readme.

## How to start

After cloning the repo, build and run with docker.

```
docker build -t sudoku-solver-app .
docker run -p 8501:8501 sudoku-solver-app
```

Open your browser and type: http://localhost:8501/

Please note that the generated computer vision solution is really simple and you should only try computer generated images. They will more or less work, real-life images most probably will fail.

# Conversation Log

## User
Hey, I'm thinking about making a Sudoku solver, formulating it as an integer program and making an easy GUI using Streamlit. Let's do it together. Are you ready?

## Assistant
I'm thrilled to help you with your Sudoku solver project! Formulating it as an integer program is a great idea, and using Streamlit for the GUI will provide an easy and interactive way to present your results. Here's a step-by-step plan to get started:

### Step 1: Understanding Sudoku
- **Objective:** Fill a 9x9 grid with digits (1 through 9) such that each column, each row, and each of the nine 3x3 subgrids contains all of the numbers from 1 to 9.

### Step 2: Formulating the Integer Program
1. **Variables:**  
   Define binary variables $( x_{ijk} )$, where $( x_{ijk} = 1 )$ if the cell in row $( i )$, column $( j )$ has the number $( k )$.

2. **Constraints:**  
   - **Single Value per Cell:**  
     - $$\sum_{k=1}^{9} x_{ijk} = 1 \quad \text{for each cell } (i, j)$$
   
   - **Uniqueness in Rows, Columns, and Subgrids:**  
     - $$\sum_{j=1}^{9} x_{ijk} = 1 \quad \text{for each row } i \text{ and number } k$$
     - $$\sum_{i=1}^{9} x_{ijk} = 1 \quad \text{for each column } j \text{ and number } k$$
     - $$\sum_{i=3p+1}^{3p+3} \sum_{j=3q+1}^{3q+3} x_{ijk} = 1 \quad \text{for each 3x3 subgrid and number } k  \text{ where} ( p, q \in {0, 1, 2} )$$

3. **Given Numbers:**
  $$x_{ijk} = 1 \quad \text{for each pre-filled cell } (i, j) \text{ with number } k$$

### Step 3: Implementing in Python
- Use libraries like `PuLP` or `Gurobi` for solving the integer program.

### Step 4: Building the GUI with Streamlit
1. **Set up Streamlit Environment:**
   - `pip install streamlit`
   - Create a new Python file: e.g., `sudoku_solver.py`

2. **Create the Interface:**
   ```python
   import streamlit as st

   def main():
       st.title("Sudoku Solver")
       st.write("Input your Sudoku puzzle:")
       # Create input fields for the 9x9 Sudoku grid...

   if __name__ == "__main__":
       main()
   ```

3. **Link the Solver with GUI:**
   - Capture user input, solve the puzzle using your integer program logic, and display the solution.

### Step 5: Testing and Enhancing
- Test with various Sudoku puzzles.
- Consider adding features like a difficulty rating or hints.

Would you like guidance on a specific part, or shall we dive deeper into any of these steps?

## User
Exactly this is how I was planning to formulate it, well done! Let's do the code with Google OR-Tools, shall we?

## Assistant
Great choice! Google OR-Tools is a powerful library for solving optimization problems, including integer programming, and it's well-suited for your Sudoku solver project. Let's build your Sudoku solver using Google OR-Tools.

### Step-by-Step Code Implementation

First, ensure you have OR-Tools installed. If not, you can install it via pip:

```bash
pip install ortools
```

Below is a basic outline to get you started on building a Sudoku solver using Google OR-Tools.

```python
from ortools.sat.python import cp_model

def create_sudoku_solver():
    # Create the model.
    model = cp_model.CpModel()

    # Create the variables.
    # x[i][j][k] will be True if cell (i, j) contains the value k+1.
    x = {}
    for i in range(9):
        for j in range(9):
            for k in range(9):
                x[i, j, k] = model.NewBoolVar(f'x[{i},{j},{k}]')

    # Add the constraints.

    # Each cell must contain exactly one value.
    for i in range(9):
        for j in range(9):
            model.AddExactlyOne(x[i, j, k] for k in range(9))

    # Each value must appear exactly once in each row.
    for i in range(9):
        for k in range(9):
            model.AddExactlyOne(x[i, j, k] for j in range(9))

    # Each value must appear exactly once in each column.
    for j in range(9):
        for k in range(9):
            model.AddExactlyOne(x[i, j, k] for i in range(9))

    # Each value must appear exactly once in each 3x3 block.
    for block_i in range(3):
        for block_j in range(3):
            for k in range(9):
                model.AddExactlyOne(
                    x[i, j, k]
                    for i in range(block_i * 3, (block_i + 1) * 3)
                    for j in range(block_j * 3, (block_j + 1) * 3)
                )
    
    return model, x

def solve_sudoku(puzzle):
    model, x = create_sudoku_solver()
    
    # Set the initial values from the puzzle into the model.
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                # puzzle[i][j] is the given number at position (i, j).
                k = puzzle[i][j] - 1
                model.Add(x[i, j, k] == 1)

    # Create the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # If a solution has been found, print it.
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        solved_grid = [[0]*9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if solver.BooleanValue(x[i, j, k]):
                        solved_grid[i][j] = k + 1
        return solved_grid
    else:
        return None

# Example Sudoku puzzle (with zeros for empty spaces)
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

solution = solve_sudoku(puzzle)
if solution:
    for row in solution:
        print(row)
else:
    print("No solution found!")
```

### Explanation

- **Model Setup:** We create a model using `cp_model.CpModel()` and define boolean variables \( x[i, j, k] \), which are true if the cell at \((i, j)\) holds the number \( k + 1 \).
- **Constraints:** We ensure each cell contains exactly one number, each number appears once per row, column, and block.
- **Input Puzzle:** Initialize the model with the known numbers from your puzzle.
- **Solving the Puzzle:** We utilize `cp_model.CpSolver()` to find a solution, storing and printing it if found.

Would you like help with integrating this into a Streamlit application, or is there another area you'd like to explore next?

## User
Let's extend this by making it a Streamlit app, where we have a dataframe input and the user can add the values then push a button to solve the Sudoku.

## Assistant
Great idea! Incorporating Streamlit will allow users to interactively input their Sudoku puzzle and solve it with the click of a button. Below is an extended version of your Sudoku solver, now integrated with a Streamlit interface.

### Streamlit App with Sudoku Solver

First, ensure you have Streamlit installed:

```bash
pip install streamlit
```

Now, you can create a new Python file, for example, `app.py`, and insert the following code:

```python
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

def create_sudoku_solver():
    model = cp_model.CpModel()
    x = {}
    for i in range(9):
        for j in range(9):
            for k in range(9):
                x[i, j, k] = model.NewBoolVar(f'x[{i},{j},{k}]')

    for i in range(9):
        for j in range(9):
            model.AddExactlyOne(x[i, j, k] for k in range(9))

    for i in range(9):
        for k in range(9):
            model.AddExactlyOne(x[i, j, k] for j in range(9))

    for j in range(9):
        for k in range(9):
            model.AddExactlyOne(x[i, j, k] for i in range(9))

    for block_i in range(3):
        for block_j in range(3):
            for k in range(9):
                model.AddExactlyOne(
                    x[i, j, k]
                    for i in range(block_i * 3, (block_i + 1) * 3)
                    for j in range(block_j * 3, (block_j + 1) * 3)
                )
    
    return model, x

def solve_sudoku(puzzle):
    model, x = create_sudoku_solver()
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != 0:
                k = puzzle[i][j] - 1
                model.Add(x[i, j, k] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        solved_grid = [[0]*9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if solver.BooleanValue(x[i, j, k]):
                        solved_grid[i][j] = k + 1
        return solved_grid
    else:
        return None

def main():
    st.title("Sudoku Solver")

    st.write("Enter your Sudoku puzzle below. Use 0 for empty cells.")

    # Initialize a 9x9 grid filled with 0.
    initial_grid = [[0]*9 for _ in range(9)]
    df = pd.DataFrame(initial_grid, columns=[f'Col {i+1}' for i in range(9)], index=[f'Row {i+1}' for i in range(9)])

    # Input grid as a dataframe.
    input_grid = st.experimental_data_editor(df, num_rows="dynamic")

    if st.button("Solve Sudoku"):
        # Convert input grid from DataFrame to list of lists.
        puzzle = input_grid.fillna(0).astype(int).values.tolist()
        # Solve the Sudoku puzzle.
        solution = solve_sudoku(puzzle)

        if solution:
            st.write("Solved Sudoku Puzzle:")
            # Display the solution.
            st.dataframe(pd.DataFrame(solution, columns=[f'Col {i+1}' for i in range(9)], index=[f'Row {i+1}' for i in range(9)]))
        else:
            st.error("No solution found!")

if __name__ == "__main__":
    main()
```

### Instructions to Run

1. Save this code in a file named `app.py`.
2. Open your terminal or command prompt.
3. Navigate to the directory where `app.py` is saved.
4. Run the Streamlit app with the following command:
   ```bash
   streamlit run app.py
   ```

### App Functionality
- **Interactive Dataframe:** Users can input their Sudoku puzzle into a 9x9 dataframe. The cells can be edited to add numbers, and 0 should represent empty cells.
- **Solve Button:** Once the puzzle is entered, clicking the "Solve Sudoku" button will trigger the solver, which uses Google OR-Tools to find the solution.
- **Result Display:** The solution is displayed in a dataframe format if found, otherwise an error message is shown.

This setup provides a user-friendly experience for solving Sudoku puzzles. Would you like to add or modify anything in this implementation?

## User
This is truly amazing, I'm so astonished by how fast and easily you solved this task. Do you think we could extend it with a camera input, where we detect the Sudoku from an image and fill the values?

## Assistant
I'm delighted to hear you're pleased with the solution! Extending the application to include camera input for detecting and solving Sudoku from an image is definitely an exciting feature. For this purpose, we can leverage image processing and computer vision techniques with libraries like OpenCV and Tesseract OCR to detect and recognize the digits.

Here's how you could go about implementing this feature:

### Step 1: Set Up Environment

First, you'll need to install additional libraries:

```bash
pip install opencv-python-headless opencv-python-contrib-headless pytesseract
```

Additionally, make sure you have Tesseract OCR installed on your system. You can find installation instructions [here for Windows](https://github.com/UB-Mannheim/tesseract/wiki), or you can use package managers like `brew` for macOS.

### Step 2: Using OpenCV and Tesseract to Detect Sudoku

```python
import cv2
import numpy as np
import pytesseract
import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

# Previously defined functions (create_sudoku_solver, solve_sudoku) go here

def process_image(uploaded_image):
    # Convert the image to a grayscale image
    gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to binarize
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Crop the puzzle from image
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = gray[y:y+h, x:x+w]

    # Divide the cropped image into 9x9 grids and analyze each cell
    grid_side = cropped.shape[0] // 9
    puzzle = []

    for i in range(9):
        row = []
        for j in range(9):
            cell = cropped[i*grid_side:(i+1)*grid_side, j*grid_side:(j+1)*grid_side]
            # Further processing like resizing can be added here
            # Perform OCR on the cell
            digit = pytesseract.image_to_string(cell, config='--psm 10 digits')
            row.append(int(digit.strip()) if digit.strip().isdigit() else 0)
        puzzle.append(row)

    return puzzle

def sudoku_from_image(image_file):
    # Load the image
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

    return process_image(image)

def main():
    st.title("Sudoku Solver with Camera Input")

    st.write("Upload an image of the Sudoku puzzle.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process the uploaded image and extract the Sudoku puzzle
        puzzle = sudoku_from_image(uploaded_file)
        st.write("Detected Sudoku Puzzle:")
        st.dataframe(pd.DataFrame(puzzle, columns=[f'Col {i+1}' for i in range(9)], index=[f'Row {i+1}' for i in range(9)]))

        if st.button("Solve Sudoku"):
            # Solve the Sudoku puzzle
            solution = solve_sudoku(puzzle)
            if solution:
                st.write("Solved Sudoku Puzzle:")
                st.dataframe(pd.DataFrame(solution, columns=[f'Col {i+1}' for i in range(9)], index=[f'Row {i+1}' for i in range(9)]))
            else:
                st.error("No solution found!")

if __name__ == "__main__":
    main()
```

### Key Points in this Implementation

- **Image Preprocessing:** Convert the uploaded image to grayscale and use adaptive thresholding to make the numbers stand out.
- **Contour Detection:** Identify and isolate the largest contour, which should correspond to the Sudoku grid, using `cv2.findContours`.
- **Grid Division:** Cut the grid into 81 small cells for digit recognition via OCR.
- **OCR with Tesseract:** Extract the digits from each cell using Tesseract OCR and prepare them for solving.

### Tips and Considerations

- **Image Quality:** The quality and angle of the image can affect OCR accuracy. Ensure the image is well-lit and the grid lines are clear.
- **OCR Configuration:** Fine-tune Tesseract's configurations for better OCR results as needed.
- **Error Handling:** Add checks and error handling to manage empty or incorrectly detected cells.

This approach leverages powerful computer vision techniques to achieve the desired functionality. If you wish to explore any further refinements or address concerns, feel free to ask!

## User
I believe with a generic OCR the detection accuracy wont be that accurate, but I do like your solution. Really impressive responses. Thank you!

## Assistant

You're absolutely right—generic OCR can sometimes struggle with accuracy, especially for specific tasks like recognizing numbers in Sudoku puzzles on varying grid qualities. For better results, you might consider a few enhancements:

-   **Digit Training:**  Use a machine learning model specifically trained on Sudoku digits for more reliable recognition. Libraries like TensorFlow or PyTorch can be used to train a custom digit recognizer based on a dataset of Sudoku puzzles.
    
-   **Pre-processing:**  Improve the image pre-processing steps by applying additional techniques such as morphological operations, edge detection, and contour approximation to clean up the image and enhance digit visibility before feeding it to the OCR.
    
-   **OCR Tuning:**  Fine-tune Tesseract parameters to optimize for clearer digit extraction by experimenting with parameters such as  `--psm`  (page segmentation mode) and  `--oem`  (OCR engine mode).
    

Feel free to reach out if you decide to explore more advanced solutions in the future or need further assistance. I'm here to help. Best of luck with your awesome project, and thank you for the kind words!


## User

Let's improve the `process_image` function. First, for the `adaptiveThreshold`, the `kernel_size` should be dynamic based on the image size. Then, after extracting the `largest_contour`, we should rectify the cropped image. Finally, let's split the `grid_size` into `grid_size_x` and `grid_size_y`.

## Assistant

To enhance the `process_image` function with dynamic thresholding, image rectification, and separate grid size dimensions, you can adjust the function as follows:

```python
import cv2
import numpy as np
import pytesseract

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
            except ValueError:
                digit_value = 0
            row.append(digit_value)
            # Draw detected digit
            if digit_value != 0:
                cv2.putText(rectified, str(digit_value), (j*grid_size_x + 10, (i+1)*grid_size_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        puzzle.append(row)

    return puzzle, rectified

def main():
    st.title("Sudoku Solver with Image Processing")

    # Sample code to simulate uploaded image handling
    uploaded_file = st.file_uploader("Upload a Sudoku image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        puzzle, rectified_visual = process_image(img)

        # Display the rectified grid with detected digits
        st.image(rectified_visual, caption="Rectified Sudoku Grid with Detected Digits", channels="GRAY")

        # Display the detected number grid
        st.write("Detected Sudoku Puzzle:")
        st.dataframe(pd.DataFrame(puzzle))

if __name__ == "__main__":
    main()
```

### Key Enhancements

1.  **Dynamic Kernel Size:**
    
    -   The kernel size for adaptive thresholding is dynamically set based on the image size and guaranteed to be odd.
2.  **Image Rectification:**
    
    -   Applied contour approximation to ensure the largest contour resembles a rectangle, assuming it's the Sudoku grid.
    -   Used perspective transformation to rectify the grid, providing a consistent view that aids digit segmentation.
3.  **Separate Grid Dimensions:**
    
    -   The rectified image is divided using separate grid heights and widths, accounting for possible distortions from the perspective transformation.

This approach should yield more robust and accurate results when extracting Sudoku puzzles from a variety of image conditions. If you need further adjustments or explanations, feel free to ask!


## User
Could you export our complete conversion in a pdf document or if not just convert it to markdown?

## Assistant
I can certainly help you convert our conversation into Markdown format! You can then easily copy this Markdown into a text editor or tool that supports Markdown to convert it into a PDF. Here's the Markdown version of our conversation:

## User
Thank you very much. Thats all really for this topic :)

## Assistant
You're welcome! I'm glad I could assist you. If you have any more questions or need help in the future, whether it's related to this project or any other topic, feel free to reach out. Have a great day and happy coding! 😊