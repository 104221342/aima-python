import sys
from itertools import product

def parse_input(filename):
    """Parse the input file to extract a generic knowledge base and query."""
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    tell_index = lines.index("TELL") + 1
    ask_index = lines.index("ASK") + 1

    tell_clauses = lines[tell_index].split(';')
    query = lines[ask_index].strip()

    facts = set()
    implications = []
    disjunctions = []
    biconditionals = []
    negations = []


    for clause in tell_clauses:
        clause = clause.strip()

        if '<=>' in clause:
            left, right = clause.split('<=>')
            biconditionals.append((left.strip(), right.strip()))
        elif '=>' in clause:
            premise, conclusion = clause.split('=>')
            implications.append((premise.strip(), conclusion.strip()))
        elif '||' in clause:
            disjunctions.append(clause.strip())
        elif clause.startswith('~'):
            negations.append(clause.strip())
        else:
            facts.add(clause)

    return facts, implications, disjunctions, biconditionals, negations, query




if __name__ == "__main__":
    filename = sys.argv[1]
    facts, implications, disjunctions, biconditionals, negations, query, variables = parse_input(filename)
    
    print("Facts:", facts)
    print("Implications:", implications)
    print("Disjunctions:", disjunctions)
    print("Biconditionals:", biconditionals)
    print("Negations:", negations)
    print("Query:", query)
    print("Variables:", variables)


