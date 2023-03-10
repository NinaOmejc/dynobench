from system import System
import numpy as np


sys_bacres = System(name="bacres",
                    state_vars=["x", "y"],
                    model=["B - x - ((x * y) / (q*x**2 + 1))",
                           "A - (x * y / (q*x**2 + 1))"],
                    model_params= {'B': 20, 'A':10, 'q':0.5},
                    init_bounds=[[4, 6], [9, 11]],
                    )

sys_barmag = System(name="barmag",
                    state_vars=["x", "y"],
                    model=["K * sin(x-y) - sin(x)",
                           "K * sin(y-x) - sin(y)"],
                    model_params={'K': 0.5},
                    init_bounds=[[1.5*np.pi, 2.5*np.pi], [1.5*np.pi, 2.5*np.pi]],
                    )


sys_glider = System(name="glider",
                    state_vars=["x", "y"],
                    model=["-D * x**2 - sin(y)",
                           "x - (cos(y)/x)"],
                    model_params= {'D': 0.05},
                    init_bounds=[[2, 8], [-3, 3]],
                    )

sys_lv = System(name="lv",
                state_vars=["x", "y"],
                model=["x * (A - x - B*y)",
                       "y * (C  - x - y)"],
                model_params= {'A': 3, 'B': 2, 'C': 2},
                init_bounds=[[1, 8], [1, 8]],
                init_cond='integers'
                )

sys_predprey = System(name="predprey",
                state_vars=["x", "y"],
                model=["x*(b - x - y/(1+x))",
                       "y*(x/(1+x) - a*y)"],
                model_params= {'a': 0.075, 'b': 4},
                init_bounds=[[2, 8], [2, 8]],
                )

sys_shearflow = System(name="shearflow",
                        state_vars=["x", "y"],
                        model=["cot(y) * cos(x)",
                               "(cos(y)**2 + A*sin(y)**2) * sin(x)"],
                        model_params= {'A': 0.1},
                        init_bounds=[[-np.pi, np.pi], [-np.pi, np.pi]],
                       )

sys_vdp = System(name="vdp",
                state_vars=["x", "y"],
                model=["y",
                       "-x + M*y*(1 - x**2)"],
                model_params= {'M': 2},
                init_bounds=[[-5, 5], [-5, 5]],
               )

sys_stl = System(name="stl",
                state_vars=["x", "y"],
                model=["a*x - w*y - x*(x**2 + y**2)",
                       "w*x + a*y - y*(x**2 + y**2)"],
                model_params= {'a': 1, 'w': 3},
                init_bounds=[[-5, 5], [-5, 5]],
               )

sys_cphase = System(name="cphase",
                    state_vars=["x", "y"],
                    aux_vars=["t"],
                    model=["A*sin(x) + B*sin(y) + W*sin(K*t) + R",
                           "C*sin(x) + D*sin(y) + E"],
                    model_params= {'A': 0.8, 'B': 0.8, 'W': 2.5, 'K': 2*np.pi*0.0015, 'R': 2,
                                   'C': 0, 'D': 0.6, 'E': 4.53},
                    init_bounds=[[-5, 5], [-5, 5]],
                   )

sys_lorenz = System(name="lorenz",
                    state_vars=["x", "y", "z"],
                    model=["S*(-x + y)",
                           "R*x - x*z - y",
                           "B*z + x*y"],
                    model_params= {'S': 10, 'R': 28, 'B': -8/3},
                    init_bounds=[[-5, 5], [-5, 5], [-5, 5]],
                   )


systems_collection = {
    sys_bacres.name: sys_bacres,
    sys_barmag.name: sys_barmag,
    sys_glider.name: sys_glider,
    sys_lv.name: sys_lv,
    sys_predprey.name: sys_predprey,
    sys_shearflow.name: sys_shearflow,
    sys_vdp.name: sys_vdp,
    sys_stl.name: sys_stl,
    sys_cphase.name: sys_cphase,
    sys_lorenz.name: sys_lorenz
}

# inits that were used to create datasets
inits_collection = {
    'bacres': [[9.01502740318998, 12.8015055992981],
               [3.48540194192317,13.0700662237925],
               [5.81391692721814,9.936214985790382],
               [5.80195472817425,9.79266054660331]],
    'barmag': [[4.26461875246862,4.76102165511013],
               [4.66923900501058,2.46443504621593],
               [4.1184890487447,1.07559768164234],
               [4.43621840643643,0.200011672644315]],
    'glider': [[5.603272452842469,1.8551204022395],
               [2.46553213651184,0.872156816895428],
               [5.35766048171177,-1.32283061311855],
               [4.51350174933492,0.346591870877256]],
    'lv':      [[1.0, 3.0],
               [4.0, 1.0],
               [8.0, 2.0],
               [3.0, 3.0]],
    'predprey': [[4.86070775030401,11.5069474930959],
               [6.58096270030409,11.4336027281584],
               [5.75343217591914,8.77853103772431],
               [5.80475500292961,11.6491125241285]],
    'shearflow': [[-1.40163422326322,-1.42574462518204],
               [-2.53129567286853,1.01617273721913],
               [1.22414434129968,-0.5745989297785539],
               [2.82882856222944,-1.462580773342]],
    'vdp': [[0.3573007050358039,0.2166979323670089],
           [0.677280004283046,0.7360147137408308],
           [0.0100185705081634,0.3559143141937729],
           [0.239954583001563,0.9276087313632232]],
    'stl': [[-1.2297032478276817,2.60244131889609],
               [0.0742153554035383,-1.0695632754414133],
               [-4.850525970889756,-1.687290101132798],
               [-4.288640073122789,-2.148638914525071]],
    'cphase': [[1.8407533177748023,-0.6600470545316153],
               [0.9027898084490892,3.029510076903197],
               [2.687451545351021,2.4793683906061474],
               [3.100309576461722,-1.3757294035867584]],
    'lorenz': [[-2.399487428702636,-0.3102402124980692,-3.5245019105708675],
               [-2.592282691932624,-4.614021504139068,-3.5245019105708675],
               [-1.6866388105859222,-1.4834823312867442,3.160401238199105],
               [-3.7706637533941088,-0.2937384967951644,-1.056978151001442]],
}