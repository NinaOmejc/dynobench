import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

class System():

    def __init__(self, name="", state_vars=[], model=[], model_params={}, **kwargs):

        self.name = name
        self.state_vars = state_vars
        self.num_vars = len(self.state_vars)
        self.model = model
        self.model_params = model_params
        self.init_bounds = kwargs.get('init_bounds', [[-5, 5]] * self.num_vars)
        self.aux_vars = kwargs.get('aux_vars', [])
        self.init_condition = kwargs.get('init_condition', 'none')

    def simulate(self, inits, time_start=0, time_end=10, time_step=0.01):
        time_span = [time_start, time_end]
        times = np.arange(time_start, time_end, time_step)

        param_keys = tuple(self.model_params.keys())
        param_vals = tuple(self.model_params.values())

        model_func = []
        for ie, expr in enumerate(self.model):
            expr_edited = check_expr(expr)
            if self.aux_vars:
                model_func.append(eval(f'lambda {", ".join(self.state_vars)}, {", ".join(self.aux_vars)}, {", ".join(param_keys)}: ' + expr_edited))
            else:
                model_func.append(eval(f'lambda {", ".join(self.state_vars)}, {", ".join(param_keys)}: ' + expr_edited))

        if self.aux_vars:
            def rhs(t, y, *param_keys):
                return [model_func[i](t, *y, *param_keys) for i in range(len(model_func))]
        else:
            def rhs(t, y, *param_keys):
                return [model_func[i](*y, *param_keys) for i in range(len(model_func))]

        simulation = solve_ivp(rhs, t_span=time_span, y0=inits, t_eval=times, args=param_vals,
                               method='LSODA', rtol=1e-12, atol=1e-12)
        return simulation


    def get_inits(self):
        if self.init_condition == 'integers':
            inits = [np.random.randint(low=i[0], high=i[1], size=(1,)) for i in self.init_bounds]
        else:
            inits = [np.random.uniform(low=i[0], high=i[1], size=(1,)) for i in self.init_bounds]
        return [i[0] for i in inits]


def check_expr(expr):
    if 'sin' in expr:
        expr = expr.replace('sin', 'np.sin')
    if 'cos' in expr:
        expr = expr.replace('cos', 'np.cos')
    if 'cot' in expr:
        expr = expr.replace('cot', 'sp.cot')
    return expr
