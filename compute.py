import pandas as pd
from typing import Callable, Dict, List, Any


class StepStatus:
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Step:
    def __init__(self, name: str, func: Callable[[Any], None], dependencies: List[str] = None):
        """
        :param name: Unique name of the step.
        :param func: Callable function that takes any input and processes it.
        :param dependencies: List of other step names that this step depends on.
        """
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.status = StepStatus.PENDING
        self.error_message = None

    def __call__(self, data: Any):
        """
        Executes the step's function on the given data and updates status.
        :param data: Generic input that can be any type (dict, list, etc.).
        """
        try:
            self.func(data)
            self.status = StepStatus.SUCCEEDED
        except Exception as e:
            self.status = StepStatus.FAILED
            self.error_message = str(e)
            print(f"Error in step '{self.name}': {self.error_message}")  # Logging error

    def reset(self):
        """
        Resets the step status to 'pending' before a new run.
        """
        self.status = StepStatus.PENDING
        self.error_message = None


class CircularDependencyError(Exception):
    """Exception raised when a circular dependency is detected."""
    pass


class Computation:
    def __init__(self):
        """Initializes an empty dictionary to store steps."""
        self.steps: Dict[str, Step] = {}

    def add_step(self, name: str, func: Callable[[Any], None], dependencies: List[str] = None):
        """Adds a step to the computation."""
        step = Step(name, func, dependencies)
        self.steps[name] = step

    def reset_steps(self):
        """Resets all steps to their initial 'pending' state before a new run."""
        for step in self.steps.values():
            step.reset()

    def _run_step(self, name: str, data: Any, visited: List[str], stack: List[str]):
        """
        Recursively runs a step and ensures its dependencies are met.
        :param name: The step to run.
        :param data: The generic input passed to the callable steps.
        :param visited: List of steps already processed.
        :param stack: Current stack to detect circular dependencies.
        """
        step = self.steps.get(name)
        if not step:
            raise ValueError(f"Step '{name}' not found")

        if name in stack:
            raise CircularDependencyError(f"Circular dependency detected: {' -> '.join(stack + [name])}")

        # Check dependencies first
        if name not in visited:
            stack.append(name)
            for dependency in step.dependencies:
                dep_step = self.steps[dependency]
                if dep_step.status == StepStatus.FAILED:
                    print(f"Step '{name}' skipped due to failed dependency '{dependency}'")
                    step.status = StepStatus.FAILED
                    step.error_message = f"Failed due to dependency '{dependency}'"
                    return
                self._run_step(dependency, data, visited, stack)
            stack.pop()

            # If all dependencies are OK, run the step
            if step.status == StepStatus.PENDING:
                step(data)

            visited.append(name)

    def __call__(self, data: Any):
        """
        Makes the Computation callable. Runs all steps on the provided data, respecting dependencies.
        :param data: Generic input passed to the computation.
        """
        self.reset_steps()  # Ensure that steps are reset before each run
        visited = []
        for step_name in self.steps:
            if self.steps[step_name].status == StepStatus.PENDING:
                self._run_step(step_name, data, visited, [])


# Define the step functions
def step_a(row):
    print(f"Running Step A on index {row.name}")
    row['new_column_1'] = row['column_1'] + 10

def step_b(row):
    print(f"Running Step B on index {row.name}")
    if row['new_column_1'] > 11:  # Artificial condition for failure
        raise ValueError("Step B failed due to invalid value in 'new_column_1'")
    row['new_column_2'] = row['new_column_1'] * 2

def step_c(row):
    print(f"Running Step C on index {row.name}")
    row['new_column_3'] = row['new_column_2'] - row['column_3']


# Create a Computation
computation = Computation()

# Add steps with dependencies
computation.add_step('A', step_a)
computation.add_step('B', step_b, dependencies=['A'])
computation.add_step('C', step_c, dependencies=['B'])

# Create a sample DataFrame
df = pd.DataFrame({
    'column_1': [1, 2, 3],  # The second row will cause Step B to fail
    'column_2': [4, 5, 6],
    'column_3': [7, 8, 9]
})

# Add placeholder columns to the DataFrame for the new values
df['new_column_1'] = None
df['new_column_2'] = None
df['new_column_3'] = None

# Apply the computation to each row
for index, row in df.iterrows():
    computation(row)

# Print the updated DataFrame
print(df)