'''

'''

INPUT1 = """
bake_vegetables chop_vegetables preheat_oven
chop_vegetables wash_vegetables
panfry_protein prepare_protein
clean_kitchen arrange_on_plate
prepare_protein
preheat_oven
serve_dish arrange_on_plate
wash_vegetables
arrange_on_plate bake_vegetables panfry_protein
"""

INPUT2 = """
bake_vegetables chop_vegetables preheat_oven
chop_vegetables wash_vegetables
panfry_protein prepare_protein arrange_on_plate
clean_kitchen arrange_on_plate
prepare_protein
preheat_oven
serve_dish arrange_on_plate
wash_vegetables
arrange_on_plate bake_vegetables panfry_protein
"""

class Task:

    def __init__(self, name: str):
        self.name = name
        self.dependencies = list["Task"]()
        self.subsequents = list["Task"]()

    @property
    def task_id(self) -> int:
        return hash(self)

    def __hash__(self):
        return hash(self.name)


class Parser:

    @staticmethod
    def parse(input: str) -> set[Task]:
        chains = [line for line in input.split("\n") if line]
        tasks = dict[str, Task]()

        for chain in chains:
            depgraph = [
                tasks.get(node, Task(node))
                for node in chain.split(" ")
                if 0 < len(node)
            ]

            tasks.update({
                task.name: task
                for task in depgraph
            })

            for target in range(len(depgraph) - 1):
                focus = depgraph[target]
                for deps in range(target + 1, len(depgraph)):
                    focus.dependencies.append(depgraph[deps])

                tasks.update({focus.name: focus})

        result = dict()
        entrypoints = list()

        for task in tasks.values():
            if len(task.dependencies) == 0:
                entrypoints.append(task)

        todos = set(tasks.values())
        dones = set()

        while todos:
            current = None

            for candidate in todos:
                nodep = len(candidate.dependencies) == 0
                allready = all(dep in dones for dep in candidate.dependencies)

                if nodep or allready:
                    current = candidate
                    print(current.name, len(todos), "Remainin...")
                    break

            if current is None:
                raise RuntimeError("No indepent tasks there...")

            todos.remove(current)
            dones.add(current)

            




            
                







Parser.parse(input=INPUT2)


