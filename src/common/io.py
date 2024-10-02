from colorama import Style


def print_attention_matrix(matrix):
    for row in matrix:
        for value in row:
            # Apply DIM style for masked cells (0.0) and NORMAL for others
            if value == 0.0:
                print(Style.DIM + f"{value:.4f}", end=" ")  # noqa: T201
            else:
                print(Style.NORMAL + f"{value:.4f}", end=" ")  # noqa: T201
        print()  # noqa: T201
