class Matrix:
    def __init__(self, matrix: list[list]) -> None:
        self.matrix = matrix
        self.check()

    @property
    def size(self) -> tuple:
        return (len(self.matrix), len(self.matrix[0]))

    def check(self) -> None:
        if len(self.matrix) == 0:
            raise ValueError("The matrix cannot be empty.")
        if len(self.matrix[0]) == 0:
            raise ValueError("All cells in the matrix cannot be empty.")
        if not all(len(row) == len(self.matrix[0]) for row in self.matrix):
            raise ValueError("All rows in the matrix must have the same length.")

    def __repr__(self) -> str:
        return f"Matrix of size {self.size} :\n{self.matrix[0]}\n{self.matrix[1]}\n{self.matrix[2]}"

    def mult_cells(self, matrix: "Matrix") -> "Matrix":
        result = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for y in range(3):
            for x in range(3):
                result.matrix[y][x] += self.matrix[y][x] * matrix.matrix[y][x]
        return result
