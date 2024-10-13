class Matrix:
    def __init__(self, matrix: list[list]) -> None:
        self.matrix = matrix
        self.size = (len(self.matrix), len(self.matrix[0]))
        self.check()

    @property
    def sum(self) -> float:
        return sum(sum(row) for row in self.matrix)

    def check(self) -> None:
        if len(self.matrix) == 0:
            raise ValueError("The matrix cannot be empty.")
        if len(self.matrix[0]) == 0:
            raise ValueError("All cells in the matrix cannot be empty.")
        if not all(len(row) == len(self.matrix[0]) for row in self.matrix):
            raise ValueError("All rows in the matrix must have the same length.")

    def __repr__(self) -> str:
        return f"Matrix of size {self.size} :\n{self.matrix[0]}\n{self.matrix[1]}\n{self.matrix[2]}"

    def convolve_by(self, kernel: "Matrix") -> int:
        if self.size != kernel.size:
            raise ValueError(
                "The two matrices must have the same size for multiplication."
            )

        result = 0
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                result += self.matrix[y][x] * kernel.matrix[y][x]

        # if kernel.sum != 0:
        #     result /= kernel.sum
        return int(result)

    def normalize(self) -> None:
        total = self.sum
        if total == 0:
            return

        for y in range(self.size[0]):
            for x in range(self.size[1]):
                self.matrix[y][x] /= total
