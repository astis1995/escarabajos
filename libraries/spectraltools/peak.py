class Peak:
    def __init__(self, x, y):
        self.x_value = x
        self.y_value = y

    def __lt__(self, other):
        return self.x_value < other.x_value

    def __str__(self):
        return f"({self.x_value}, {self.y_value})"

    def __repr__(self):
        return f"({self.x_value}, {self.y_value})"

    def get_x(self):
        return self.x_value

    def get_y(self):
        return self.y_value