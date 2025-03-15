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
    solver.parameters.log_search_progress = True
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
    
def get_sample_puzzle():
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
    return puzzle

if __name__ == "__main__":
    # Example Sudoku puzzle (with zeros for empty spaces)
    puzzle = get_sample_puzzle()

    solution = solve_sudoku(puzzle)
    if solution:
        for row in solution:
            print(row)
    else:
        print("No solution found!")