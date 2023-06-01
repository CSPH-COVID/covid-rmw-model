import copy
import networkx as nx
import numpy as np
import sympy
import tensorflow as tf
import time
import datetime as dt
from itertools import product
from functools import cache
from tf_data import ModelParameters


class TimeVaryingParameter:

    def __init__(self, indices, values):
        self.val = None
        self.idx = None
        # Indices and values for the "value" variables
        if values is None:
            indices = [0]
            values = [1.0]
        self.n = len(indices)
        self.range_n = tf.range(self.n)
        self.update_vals(idx=indices, val=values)

    def update_vals(self, idx, val):
        if self.val is None or len(val) != self.val.shape[0]:
            #print("Shape mismatch when updating variable, tf.Variable will be recreated...")
            self.idx = tf.Variable(initial_value=idx, dtype=tf.int32)
            self.val = tf.Variable(initial_value=val, dtype=tf.float32)
        else:
            self.idx.assign(idx)
            self.val.assign(val)

    def __getitem__(self, t):
        return self.val[tf.reduce_max(self.range_n[(self.idx - t) <= 0])]


class Model:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("all")
        self.cmpts_def = {'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                          'age': ['0-17', '18-64', '65+'],
                          'vacc': ['none', 'shot1', 'shot2', 'booster1', 'booster23'],
                          'variant': ['none', 'wildtype', 'alpha', 'delta', 'omicron', 'ba2', 'ba45', 'bq', 'xbb'],
                          'immun': ['low', 'medium', 'high']}

        self.level_to_idx = {key: {val: idx for idx, val in enumerate(vals)} for key, vals in self.cmpts_def.items()}

        self.cmpts_lookup = {t: i for i, t in enumerate(product(*self.cmpts_def.values()))}

        self.cmpts_shape = tuple(len(v) for v in self.cmpts_def.values())

        self.cmpts = {}

        self.start_date = dt.datetime.strptime("2020-01-01","%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime("2023-06-01","%Y-%m-%d").date()

        self.param_t = None
        self.param_v = None
        self.param_row_idx = None

        self.mults_t = None
        self.mults_v = None
        self.mults_row_idx = None

        self.flows = []
        self.flow_idcs = []
        self.n_flows = 0

        self.scale = []
        self.scale_idcs = []

    @property
    def n_cmpts(self):
        return np.prod(self.cmpts_shape)

    @property
    def n_edges(self):
        return self.graph.number_of_edges()

    def make_progress_message(self, msg, i, tot):
        n_d = len(str(tot))
        fmt_idcs = "({:0%dd}/{:0%dd})" % (n_d, n_d)
        fmt_str = ("{} " + fmt_idcs).format(msg, i, tot)
        return fmt_str

    def cmpt_to_idx(self, cmpt):
        return np.squeeze(np.ravel_multi_index([[lvl_dict[elem]] for elem,lvl_dict in zip(cmpt,self.level_to_idx.values())],dims=self.cmpts_shape)).item()

    def date_to_t(self, date):
        """Convert a date (string or date object) to t, number of days since model start date.

        Args:
            date: either a string in the format 'YYYY-MM-DD' or a date object.

        Returns: integer t, number of days since model start date.

        """
        if isinstance(date, str):
            return (dt.datetime.strptime(date, "%Y-%m-%d").date() - self.start_date).days
        else:
            return (date - self.start_date).days

    def get_ids(self, **kwargs):
        bad_keys = set(kwargs.keys()) - set(self.cmpts_def.keys())
        if len(bad_keys) != 0:
            raise RuntimeError(f"Key(s): {bad_keys} do not refer to a compartment level: {set(self.cmpts_def.keys())}")
        used_cmpts = []
        for level in self.cmpts_def.keys():
            if level in kwargs:
                arg = kwargs[level]
                used_cmpts.append(arg if type(arg) == list else [arg])
            else:
                used_cmpts.append(self.cmpts_def[level])
        return [{level: value for level, value in zip(self.cmpts_def.keys(), cmpt)} for cmpt in product(*used_cmpts)]

    def add_flow(self, from_cmpts: dict, to_cmpts: dict, scale_cmpts=None, from_node_coef=None, to_node_coef=None, edge_coef=None):
        # All 'from' compartments which participate
        from_ids = self.get_ids(**from_cmpts)

        to_ids = copy.deepcopy(from_ids)
        for to_id in to_ids:
            to_id.update(to_cmpts)

        #from_idx = [self.cmpts_lookup[tuple(val.values())] for val in from_ids]
        #to_idx = [self.cmpts_lookup[tuple(val.values())] for val in to_ids]

        #self.graph.add_edges_from([(tuple(u.values()),tuple(v.values())) for u,v in zip(from_ids,to_ids)])

        from_node_expr = sympy.parse_expr(from_node_coef if from_node_coef is not None else "1")
        to_node_expr = sympy.parse_expr(to_node_coef if to_node_coef is not None else "1")
        edge_coef = sympy.parse_expr(edge_coef if edge_coef is not None else "1")

        data_dict = {"flow": {"from_coef": from_node_expr,
                               "to_coef": to_node_expr,
                               "edge_coef": edge_coef,
                               "scale_cmpts_coef": scale_cmpts}}

        self.graph.add_edges_from([(tuple(u.values()), tuple(v.values()), data_dict) for u, v in zip(from_ids, to_ids)])

    @staticmethod
    def parse_expr_string(string):
        expr = sympy.parse_expr(s=string)
        return expr

    @staticmethod
    @cache
    def get_tf_constant(val):
        return tf.constant(val,dtype=tf.float32)

    @staticmethod
    def get_tf_func(expr):
        if not expr.free_symbols:
            return Model.get_tf_constant(float(expr))
        args = [str(a) for a in expr.free_symbols]
        return sympy.lambdify(args=args, expr=expr, modules="tensorflow")

    def build_ode_flows(self):
        self.graph.add_nodes_from(self.cmpts_lookup.keys())
        # Vaccination
        self.add_flow(from_cmpts={"seir": ["S", "E", "A"], "vacc": "none"},
                      to_cmpts={"vacc": "shot1", "immun": "medium"},
                      from_node_coef="shot1_per_available * shot1_ve")
        self.add_flow(from_cmpts={"seir": ["S", "E", "A"], "vacc": "none"},
                      to_cmpts={"vacc": "shot1", "immun": "low"},
                      from_node_coef="shot1_per_available * (1-shot1_ve)")
        for (from_shot, to_shot) in [('shot1', 'shot2'), ('shot2', 'booster1'), ('booster1', 'booster23')]:
            for immun in self.cmpts_def["immun"]:
                if immun == "low":
                    self.add_flow(from_cmpts={"seir": ["S", "E", "A"], "vacc": from_shot, "immun": immun},
                                  to_cmpts={"vacc": to_shot, "immun": "high"},
                                  from_node_coef=f"{to_shot}_per_available * {to_shot}_ve")
                    self.add_flow(from_cmpts={"seir": ["S", "E", "A"], "vacc": from_shot, "immun": immun},
                                  to_cmpts={"vacc": to_shot, "immun": "low"},
                                  from_node_coef=f"{to_shot}_per_available * (1-{to_shot}_ve)")
                else:
                    self.add_flow(from_cmpts={"seir": ["S", "E", "A"], "vacc": from_shot, "immun": immun},
                                  to_cmpts={"vacc": to_shot, "immun": "high"},
                                  from_node_coef=f"{to_shot}_per_available")

        # Exposure
        for variant in self.cmpts_def["variant"]:
            if variant == "none":
                continue
            self.add_flow(from_cmpts={'seir': 'S'},
                          to_cmpts={'seir': 'E', 'variant': variant},
                          to_node_coef='lamb * betta',
                          from_node_coef=f'(1 - immunity) * kappa / region_pop',
                          scale_cmpts={'seir': 'I', 'variant': variant})
            self.add_flow(from_cmpts={'seir': 'S'},
                          to_cmpts={'seir': 'E', 'variant': variant},
                          to_node_coef='betta',
                          from_node_coef=f'(1 - immunity) * kappa / region_pop',
                          scale_cmpts={'seir': 'A', 'variant': variant})
            self.add_flow(from_cmpts={'seir': 'S'},
                          to_cmpts={'seir': 'E', 'variant': variant},
                          to_node_coef="lamb * betta",
                          from_node_coef=f'immunity * kappa / region_pop',
                          edge_coef='immune_escape',
                          scale_cmpts={'seir': 'I', 'variant': variant})
            self.add_flow(from_cmpts={'seir': 'S'},
                          to_cmpts={'seir': 'E', 'variant': variant},
                          to_node_coef="betta",
                          from_node_coef=f'immunity * kappa / region_pop',
                          edge_coef='immune_escape',
                          scale_cmpts={'seir': 'A', 'variant': variant})
        # Disease Progression
        self.add_flow(from_cmpts={"seir": "E"},
                      to_cmpts={"seir": "I"},
                      to_node_coef="1 / alpha * pS")
        self.add_flow(from_cmpts={"seir": "E"},
                      to_cmpts={"seir": "A"},
                      to_node_coef="1 / alpha * (1 - pS)")

        self.add_flow(from_cmpts={"seir": "I"},
                      to_cmpts={"seir": "Ih"},
                      to_node_coef="gamm * hosp * (1 - severe_immunity) * (1 - mab_prev - pax_prev)")
        self.add_flow(from_cmpts={"seir": "I"},
                      to_cmpts={"seir": "Ih"},
                      to_node_coef="gamm * hosp * (1 - severe_immunity) * mab_prev * mab_hosp_adj")
        self.add_flow(from_cmpts={"seir": "I"},
                      to_cmpts={"seir": "Ih"},
                      to_node_coef="gamm * hosp * (1 - severe_immunity) * pax_prev * pax_hosp_adj")

        # Disease Termination
        for variant in self.cmpts_def["variant"]:
            if variant == "none":
                continue
            self.add_flow(from_cmpts={"seir": "I", "variant": variant},
                          to_cmpts={"seir": "S", "immun": "high"},
                          to_node_coef="gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)")
            self.add_flow(from_cmpts={"seir": "I", "variant": variant},
                          to_cmpts={"seir": "S"},
                          to_node_coef="gamm * (1 - hosp - dnh) * priorinf_fail_rate")

            for prior_immunity in self.cmpts_def["immun"]:
                resulting_immunity = "medium" if prior_immunity == "low" else "high"
                self.add_flow(from_cmpts={"seir": "A", "variant": variant, "immun": prior_immunity},
                              to_cmpts={"seir": "S", "immun":resulting_immunity},
                              to_node_coef="gamm * (1 - priorinf_fail_rate)")
                self.add_flow(from_cmpts={"seir": "A", "variant": variant, "immun": prior_immunity},
                              to_cmpts={"seir": "S"},
                              to_node_coef="gamm * priorinf_fail_rate")

            self.add_flow(from_cmpts={'seir': 'Ih', 'variant': variant},
                          to_cmpts={'seir': 'S', 'immun': 'high'},
                          to_node_coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate) * (1-mab_prev)')
            self.add_flow(from_cmpts={'seir': 'Ih', 'variant': variant},
                          to_cmpts={'seir': 'S'},
                          to_node_coef='1 / hlos * (1 - dh) * priorinf_fail_rate * (1-mab_prev)')
            self.add_flow(from_cmpts={'seir': 'Ih', 'variant': variant},
                          to_cmpts={'seir': 'S', 'immun': 'high'},
                          to_node_coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate) * mab_prev')
            self.add_flow(from_cmpts={'seir': 'Ih', 'variant': variant},
                          to_cmpts={'seir': 'S'},
                          to_node_coef='1 / hlos * (1 - dh) * priorinf_fail_rate * mab_prev')

            self.add_flow(from_cmpts={'seir': 'I', 'variant': variant},
                          to_cmpts={'seir': 'D'},
                          to_node_coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flow(from_cmpts={'seir': 'Ih', 'variant': variant},
                          to_cmpts={'seir': 'D'},
                          to_node_coef='1 / hlos * dh')
        # Immunity Decay
        for seir in self.cmpts_def["seir"]:
            if seir == "D":
                continue

            self.add_flow(from_cmpts={"seir": seir, "immun": "high"},
                          to_cmpts={"immun": "medium"},
                          to_node_coef="1 / imm_decay_high_medium")

            self.add_flow(from_cmpts={"seir": seir, "immun": "medium"},
                          to_cmpts={"immun": "low"},
                          to_node_coef="1 / imm_decay_medium_low")

        print("All flows built.")

    def load_parameters_test(self, params):
        p_vals = []
        p_rows = []
        p_cols_t = []
        idx = 0
        for param in params:
            if ("from_attrs" in param or "attrs" in param):
                if "vals" in param:
                    # Node attribute
                    if "attrs" in param:
                        param_nodes = ["all"] if param["attrs"] is None else self.get_ids(**param["attrs"])
                        param_vals = param["vals"]
                        param_row = [idx] * len(param_vals.keys())
                        param_ts = [self.date_to_t(d) for d in param_vals.keys()]
                        param_v = [v for v in param_vals.values()]
                elif "mults" in param:
                    pass
                # Add to arrays
                p_rows.extend(param_row)
                p_cols_t.extend(param_ts)
                p_vals.extend(param_v)
            idx += 1
        ragged_t = tf.RaggedTensor.from_value_rowids(p_cols_t, p_rows)
        ragged_v = tf.RaggedTensor.from_value_rowids(p_vals, p_rows)
        return params

    def load_parameters(self, params):
        # Ragged Tensor for Parameters
        p_i = 0
        p_vals = []
        p_rows = []
        p_cols_t = []
        # Ragged Tensor for Multipliers
        # For convenience, I hardcode a multiplier of 1.0 which gets used if not multiplier is active.
        m_i = 1
        m_vals = [1.0]
        m_rows = [0]
        m_cols_t = [0]

        for param in params:
            param_name = param["param"]
                # Value parameter
            if "vals" in param:
                param_vals = param["vals"]
                param_c_t = [self.date_to_t(k) for k in param_vals.keys()]
                param_v = [v for v in param_vals.values()]
                # Values are the actual parameter values
                p_vals.extend(param_v)
                # Rows select which parameter
                p_rows.extend([p_i] * len(param_v))
                # Columns select the time t when we want the parameter
                p_cols_t.extend(param_c_t)

                if "from_attrs" in param:
                    from_attrs = param["from_attrs"]
                    to_attrs = param["to_attrs"]

                    from_nodes = ["all"] if from_attrs is None else [tuple(k.values()) for k in self.get_ids(**from_attrs)]

                    for from_node in from_nodes:
                        if from_node == "all":
                            to_nodes = ["all"] if to_attrs is None else self.get_ids(**to_attrs)
                        else:
                            # Don't copy the seir attribute.
                            to_node_attrs = {k:v for k,v in zip(self.cmpts_def.keys(),from_node) if k != "seir"}
                            to_node_attrs.update(to_attrs)
                            to_nodes = [tuple(k.values()) for k in self.get_ids(**to_node_attrs)]
                        for to_node in to_nodes:
                            if self.graph.has_edge(from_node,to_node):
                                for pk, d in self.graph[from_node][to_node].items():
                                    if "params" not in d:
                                        d["params"] = {}
                                    d["params"].update({param_name: p_i})
                            elif from_node == "all" or to_node == "all":
                                self.graph.add_edge(from_node,to_node,**{"params": {param_name: p_i}})
                                #self.graph[from_node][to_node].update({param_name: tvp})
                elif "attrs" in param:
                    attrs = param["attrs"]
                    nodes = ["all"] if attrs is None else [tuple(k.values()) for k in self.get_ids(**attrs)]
                    #tvp = TimeVaryingParameter(indices=param_idx, values=param_v)
                    for node in nodes:
                        node_dict = self.graph.nodes[node]
                        if "params" not in node_dict:
                            node_dict["params"] = {}
                        node_dict["params"].update({param_name: p_i})
                else:
                    raise RuntimeError(f"Parameter {param_name} must have either 'from_attrs'/'to_attrs' keys or an 'attrs' key!")
                p_i += 1
            elif "mults" in param:
                param_mults = param["mults"]
                param_mults_c_t = [self.date_to_t(k) for k in param_mults.keys()]
                param_mults_v = [v for v in param_mults.values()]
                # Values are the actual multiplier values
                m_vals.extend(param_mults_v)
                # Rows select which multiplier to use
                m_rows.extend([m_i] * len(param_mults_v))
                # Columns select the time t when we want the multiplier
                m_cols_t.extend(param_mults_c_t)
                # Find the existing param associated with this multiplier
                if "from_attrs" in param:
                    from_attrs = param["from_attrs"]
                    to_attrs = param["to_attrs"]

                    from_nodes = ["all"] if from_attrs is None else [tuple(k.values()) for k in self.get_ids(**from_attrs)]
                    for from_node in from_nodes:
                        if from_node == "all":
                            to_nodes = ["all"] if to_attrs is None else self.get_ids(**to_attrs)
                        else:
                            to_node_attrs = {k: v for k, v in zip(self.cmpts_def.keys(), from_node) if k != "seir"}
                            to_node_attrs.update(to_attrs)
                            to_nodes = [tuple(k.values()) for k in self.get_ids(**to_node_attrs)]
                        for to_node in to_nodes:
                            if self.graph.has_edge(from_node, to_node):
                                for mk, d in self.graph[from_node][to_node].items():
                                    if "mults" not in d:
                                        d["mults"] = {}
                                    d["mults"].update({param_name: m_i})
                            elif from_node == "all" or to_node == "all":
                                self.graph.add_edge(from_node, to_node, **{"params": {param_name: m_i}})
                elif "attrs" in param:
                    attrs = param["attrs"]
                    nodes = ["all"] if attrs is None else [tuple(k.values()) for k in self.get_ids(**attrs)]
                    for node in nodes:
                        node_dict = self.graph.nodes[node]
                        if "mults" not in node_dict:
                            node_dict["mults"] = {}
                        node_dict["mults"].update({param_name: m_i})
                m_i += 1
            else:
                raise RuntimeError(f"Parameter {param_name} must have either a 'vals' or 'mults' key!")
        self.param_t = tf.RaggedTensor.from_value_rowids(p_cols_t, p_rows)
        self.param_v = tf.RaggedTensor.from_value_rowids(p_vals, p_rows)
        self.param_row_idx = tf.range(self.param_t.shape[0])

        self.mults_t = tf.RaggedTensor.from_value_rowids(m_cols_t, m_rows)
        self.mults_v = tf.RaggedTensor.from_value_rowids(m_vals, m_rows)
        self.mults_row_idx = tf.range(self.mults_t.shape[0])
        print("Finished assigning attributes.")

    def params_at_t(self, t):
        idcs = tf.stack([self.param_row_idx, tf.reduce_sum(tf.cast(self.param_t <= t, tf.int32), axis=1) - 1], axis=1)
        tf.debugging.assert_non_negative(idcs)
        return tf.gather_nd(self.param_v, idcs)

    def mults_at_t(self, t):
        idcs = tf.stack([self.mults_row_idx, tf.reduce_sum(tf.cast(self.mults_t <= t, tf.int32), axis=1) - 1], axis=1)
        tf.debugging.assert_non_negative(idcs)
        return tf.gather_nd(self.mults_v, idcs)

    def create_tf_funcs(self):
        i = 0
        f = []
        s = []
        s_i = []
        n_edges = self.graph.number_of_edges()
        for from_node, to_node, edge_idx, edge_dict in self.graph.edges(data=True,keys=True):
            print(f"Parsed {i:06d} of {n_edges:06d} edges...",end="\r" if i != n_edges else "\n")
            if "flow" in edge_dict:
                u = self.cmpt_to_idx(from_node)
                v = self.cmpt_to_idx(to_node)
                # This edge has a flow
                all_expr_vars = {}
                coefs = edge_dict["flow"]

                edge_dict["tensors"] = {}

                all_expr_vars.update(self.get_expr_vars(expr=coefs["from_coef"],node=from_node))
                all_expr_vars.update(self.get_expr_vars(expr=coefs["to_coef"],node=to_node))
                all_expr_vars.update(self.get_expr_vars(expr=coefs["edge_coef"],edge_d=edge_dict))

                tf_func = self.get_tf_func(coefs["from_coef"] * coefs["to_coef"] * coefs["edge_coef"])
                p_args,m_args = zip(*list(all_expr_vars.values()))
                scale_idcs = [] if coefs["scale_cmpts_coef"] is None else [self.cmpt_to_idx(c) for c in [tuple(v.values()) for v in self.get_ids(**coefs["scale_cmpts_coef"])]]
                #edge_dict["tensors"]["func"] = tf_func
                #edge_dict["tensors"]["args"] = list(all_expr_vars.values())
                #edge_dict["tensors"]["scale_idcs"] = None if coefs["scale_cmpts_coef"] is None else [self.cmpt_to_idx(c) for c in [tuple(v.values()) for v in self.get_ids(**coefs["scale_cmpts_coef"])]]

                f.append((u, v, tf_func, p_args, m_args))
                s_i.extend([[i,v] for v in scale_idcs])
                s.extend([1]*len(scale_idcs))
                i += 1
        self.flows = f
        self.n_flows = tf.constant(len(f))
        self.scale = tf.SparseTensor(indices=s_i,values=s,dense_shape=(len(f),self.n_cmpts))

    def create_tf_funcs_2(self):
        i=1
        for node,node_params in self.graph.nodes(data=True):
            for _, v, out_edge_idx, out_edge_dict in self.graph.out_edges(node,data=True,keys=True):
                if "flow" not in out_edge_dict:
                    continue

    def get_expr_vars(self, expr, node=None, edge_d=None):
        expr_vars = {}
        expr_symbols = [str(x) for x in expr.free_symbols]
        if node is not None:
            for exp_var in expr_symbols:
                # Search for parameter source
                param_v = None
                for source in [node,"all"]:
                    if "params" in self.graph.nodes[source] and exp_var in self.graph.nodes[source]["params"]:
                        param_v = self.graph.nodes[source]["params"][exp_var]
                        break
                if param_v is None:
                    raise RuntimeError(f"Parameter {exp_var} not defined for compartment {node}!")

                # Multiplier just defaults to 0 (the hardcoded 1.0 index) if it is not defined.
                mult_v = 0
                for source in [node,"all"]:
                    if "mults" in self.graph.nodes[source] and exp_var in self.graph.nodes[source]["mults"]:
                        mult_v = self.graph.nodes[source]["mults"][exp_var]
                        break
                expr_vars[exp_var] = (param_v, mult_v)
        elif edge_d is not None:
            for exp_var in expr_symbols:
                if "params" in edge_d and exp_var in edge_d["params"]:
                    param_v = edge_d["params"][exp_var]
                elif exp_var in self.graph["all"]["all"][0]["params"]:
                    param_v = self.graph["all"]["all"][0]["params"][exp_var]
                else:
                    raise RuntimeError("Unable to find parameter!")
                mult_v = 0
                if "mults" in edge_d and exp_var in edge_d["mults"]:
                    mult_v = edge_d["mults"][exp_var]
                elif "mults" in self.graph["all"]["all"][0] and exp_var in self.graph["all"]["all"][0]["mults"]:
                    mult_v = self.graph["all"]["all"][0]["mults"][exp_var]
                expr_vars[exp_var] = (param_v, mult_v)

        return expr_vars

    def get_flow_matrix(self, t):
        idcs = []
        vals = []
        params_t = self.params_at_t(t)
        mults_t = self.mults_at_t(t)
        for from_idx, to_idx, func, p_args, m_args in self.flows:
            params = tf.gather(params_t,p_args)
            mults = tf.gather(mults_t, m_args)
            params_mults = tf.multiply(params,mults)
            val = func(*tf.unstack(params_mults))
            vals.extend([-val,val])
            idcs.extend([(from_idx,from_idx),(to_idx,from_idx)])
        return tf.scatter_nd(idcs,vals,[self.n_cmpts]*2)

    @tf.function
    def ode_with_matmul(self, t, y):
        """
        :param t: Tensor representing the time T at which to compute dy.
        :param y: Tensor representing the state vector at time T.
        :param param_v: RaggedTensor representing the values for all parameters.
        :param param_t: RaggedTensor representing the times T where each parameter changes.
        :param mults_v: RaggedTensor representing the values for all multipliers.
        :param mults_t: RaggedTensor representing the times T where each multiplier changes.
        :return: a Tensor representing the dy at time T.
        """
        flow_matrix = self.get_flow_matrix(t)
        return tf.matmul(flow_matrix,tf.expand_dims(y,axis=1))

    @tf.function
    def ode(self, t, y):
        dy = list([0]*len(y))
        params_t = self.params_at_t(t)
        mults_t = self.mults_at_t(t)
        for from_idx, to_idx, func, p_args, m_args in self.flows:
            params = tf.gather(params_t, p_args)
            mults = tf.gather(mults_t, m_args)
            params_mults = tf.multiply(params,mults)
            func_val = func(*tf.unstack(params_mults))

            dy[from_idx] -= func_val * y[from_idx]
            dy[to_idx] += func_val * y[from_idx]
        return dy


if __name__ == "__main__":
    start_date = dt.datetime.strptime("2020-01-01","%Y-%m-%d").date()
    end_date = dt.datetime.strptime("2023-06-30","%Y-%m-%d").date()
    region = "con"

    model_parameters = ModelParameters(start_date=start_date, end_date=end_date, region=region)
    model_parameters.load_all_params()
    # hosp_data = get_hosps()
    model = Model()
    model.build_ode_flows()
    model.load_parameters(params=model_parameters.params)
    model.create_tf_funcs()
    #logdir = "tf_model/log"
    #writer = tf.summary.create_file_writer(logdir)
    #tf.summary.trace_on(graph=True, profiler=True)

    # Test Matmul version
    elapsed_matmul = []
    for i in range(1000):
        s_t = time.perf_counter()
        t0 = model.ode_with_matmul(tf.constant(i), tf.zeros(shape=np.prod(model.cmpts_shape)))
        #t1 = model.ode(tf.constant(i), tf.zeros(shape=np.prod(model.cmpts_shape)))
        elapsed = time.perf_counter() - s_t
        print(f"Finished in {elapsed:03f} seconds.")
        elapsed_matmul.append(elapsed)
    print(f"Matmul version Median: {np.median(elapsed_matmul):03f}s Std: +/-{np.std(elapsed_matmul):03f}s")
    # Non-matmul version
    elapsed_nmm = []
    for i in range(1000):
        s_t = time.perf_counter()
        t0 = model.ode(tf.constant(i), tf.zeros(shape=np.prod(model.cmpts_shape)))
        elapsed_n = time.perf_counter() - s_t
        print(f"Finished in {elapsed_n:03f} seconds.")
        elapsed_nmm.append(elapsed_n)
    print(f"Non-matmul version Median: {np.median(elapsed_nmm):03f}s Std: +/-{np.std(elapsed_nmm):03f}s")
    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="ode_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)
    print("Done!")
