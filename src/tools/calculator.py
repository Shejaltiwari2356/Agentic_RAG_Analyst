class MathTool:
    def calculate(self, expression: str) -> str:
        """Evaluates mathematical expressions (e.g., '(109158 / 96169) - 1')."""
        try:
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error: {str(e)}"