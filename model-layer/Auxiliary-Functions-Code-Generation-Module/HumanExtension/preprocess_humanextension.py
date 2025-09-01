import ast
import json
import pkgutil
import re

from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape



environment = Environment(trim_blocks=True,
                          lstrip_blocks=False,
                          keep_trailing_newline=True,
                          loader=FileSystemLoader("src/transformer_analysis/templates"),
                          autoescape=select_autoescape())

template = environment.get_template("humanextension.j2")

def create_prompt(row):

    def make_docstring_node(docstring: str) -> ast.Expr:
        docstring = docstring.replace('\n', '\n    ')
        return ast.Expr(value=ast.Constant(docstring))

    def create_code(function_dict: dict, no_body: bool) -> str:
        code = function_dict['template'].format(**{x: x for x in function_dict['placeholders']})
        docstring_str = function_dict['docstring']['description'] + '\n\n' + '\n'.join(function_dict['docstring']['examples'])
        node = ast.parse(code).body[0]
        if no_body:
            node.body = [make_docstring_node(docstring_str)]
        else:
            node.body = [make_docstring_node(docstring_str)] + node.body
        return ast.unparse(node)

    auxiliary_function_code = create_code(row['auxiliary_function'], False)
    target_function_code = create_code(row['target_function'], True)

    data = {
        "imports": row['imports'],
        "auxiliary_function_code": auxiliary_function_code,
        "target_function_code": target_function_code 
    }
    prompt = template.render(**data)
    return prompt


def parse_imports(code: str):
    nodes = ast.parse(code).body
    result = []
    for node in nodes:
        if isinstance(node, ast.Import):
            result.append((None, tuple(name.name for name in node.names)))
        elif isinstance(node, ast.ImportFrom):
            result.append((node.module, tuple(name.name for name in node.names)))
        else:
            raise ValueError()
    return result


BUILTIN_NAMES = ['abs', 'aiter', 'all', 'anext', 'any', 'ascii', 'bin', 'bool', 'breakpoint',
                 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
                 'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
                 'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash',
                 'help', 'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter',
                 'len', 'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
                 'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
                 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
                 'super', 'tuple', 'type', 'vars', 'zip']

BUILTIN_NAMES += [module_info.name for module_info in pkgutil.iter_modules()]

class EncloseVariable(ast.NodeTransformer):
    def __init__(self, arg_names, names):
        super().__init__()
        self.arg_names = arg_names
        self.names = names
        self.captured_variables = set()

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        self.captured_variables.add(node.name)
        node.name = f"{{{{ {node.name} }}}}"
        for arg in node.args.args:
            self.captured_variables.add(arg.arg)
            arg.arg = f"{{{{ {arg.arg} }}}}"
        return node

    def visit_Name(self, node):
        self.generic_visit(node)
        if (node.id in self.arg_names) or (node.id not in self.names):
            self.captured_variables.add(node.id)
            new_node = ast.Name(id=f"{{{{ {node.id} }}}}", ctx=node.ctx)
        else:
            new_node = node
        return new_node


def parse_function(imports: str, code: str):
    node = ast.parse(code).body[0]

    # get function name
    name = node.name

    # get argument name
    arguments = [arg.arg for arg in node.args.args]

    # get docstring
    docstring = ast.get_docstring(node)
    if docstring is not None:
        docstring = {
            'description': docstring[:docstring.find('>>>')].strip() if '>>>' in docstring else docstring.strip(),
            'examples': re.findall(r'>>>[^\n]*\n[^\n]*(?=\n|$)', docstring, flags=re.M)
        }
        _description = docstring['description'].lower()
        if _description.endswith('It must be implemented like this:'):
            docstring['description'] = docstring['description'][:-len('It must be implemented like this:')].strip()
        if _description.endswith('[input/output] samples:'):
            docstring['description'] = docstring['description'][:-len('[input/output] samples:')].strip()
        elif _description.endswith('[inpt/output] samples:'):
            docstring['description'] = docstring['description'][:-len('[inpt/output] samples:')].strip()
        elif _description.endswith('for examples:'):
            docstring['description'] = docstring['description'][:-len('for examples:')].strip()
        elif _description.endswith('for example:'):
            docstring['description'] = docstring['description'][:-len('for example:')].strip()
        elif _description.endswith('for examble:'):
            docstring['description'] = docstring['description'][:-len('for examble:')].strip()
        elif _description.endswith('example 1:'):
            docstring['description'] = docstring['description'][:-len('example 1:')].strip()
        elif _description.endswith('examples:'):
            docstring['description'] = docstring['description'][:-len('examples:')].strip()
        elif _description.endswith('examples'):
            docstring['description'] = docstring['description'][:-len('examples')].strip()
        elif _description.endswith('example :'):
            docstring['description'] = docstring['description'][:-len('example :')].strip()
        elif _description.endswith('example:'):
            docstring['description'] = docstring['description'][:-len('example:')].strip()
        elif _description.endswith('example'):
            docstring['description'] = docstring['description'][:-len('example')].strip()
            
        node.body = node.body[1:]

    # get template
    imported_names = [keyword for path, keywords in parse_imports(imports) for keyword in keywords]

    transformer = EncloseVariable(arg_names=arguments, names=BUILTIN_NAMES + imported_names)
    new_node = transformer.visit(node)
    template = ast.unparse(new_node)

    placeholders = list(transformer.captured_variables)

    return {"name": name,
            "arguments": arguments,
            "docstring": docstring,
            "template": template,
            "placeholders": placeholders} 


dataset = load_dataset("sh0416/humanextension", split="test")

dataset = [
    {
        "imports": parse_imports(row["imports"]),
        "auxiliary_function": parse_function(row["imports"], row['function1_human']),
        "target_function": parse_function(row["imports"], row["function2_human"]),
        "tests": row["tests"]
    }
    for row in dataset
]

with open('data.json', 'w') as f:
    json.dump(dataset, f, indent=2)