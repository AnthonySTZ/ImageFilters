class Matrix3:
    def __init__(self, matrix: list[list]) -> None:
        self.matrix = matrix

    def __repr__(self) -> str:
        return f"Matrix3:\n{self.matrix[0]}\n{self.matrix[1]}\n{self.matrix[2]}"
