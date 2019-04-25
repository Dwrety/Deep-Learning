from data_utils import *



def greedy_search(preds):
    # probs: FloatTensor (B, L, C)
    out = []
    for pred in preds:
        s = []
        for step in pred:
            #             idx = torch.multinomial(step, 1)[0]
            idx = step.max(0)[1][0]
            c = index2char[idx]
            s.append(c)
            if c == '<eos>':
                break
        out.append("".join(s))
    return out


class BeamTree(object):
    def __init__(self, parent, state, value, cost, things):
        super(BeamTree, self).__init__()

        self.parent = parent
        self.state = state
        self.value = value
        self.accumulated_cost = parent.accumulated_cost + cost if parent is not None else cost
        self.depth = 1 if parent is None else parent.depth + 1

        if things is not None:
            self.things = things
        self._sequence = None

    def construct_tree(self):
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def get_seq_values(self):
        return [node.value for node in self.construct_tree()]

    def get_seq_append(self):
        return [node.things for node in self.construct_tree()]


# only support batch_size 1 currently
def beam_search_decoder(las_model, generator, terminate_idx, input_x, seq_len, beam_width=5, top_k=1, max_length=250):  
    init_state, init_value = las_model(input_x, seq_len)
    next_nodes = [BeamTree(parent=None, state=init_state, value=init_value, cost=0.0, things=None)]
    K = []

    for step in range(max_length):
        children = []
        for node in next_nodes:
            if (step != 0 and node.value == terminate_idx) or step == max_length - 1:
                K.append(node)

            else:
                children.append(node)
        if not children or len(K) >= top_k:
            break

        y_ = [node.value for node in children]
        state_ = [node.state for node in children]
        state_new, pred, things_new = generator(input_x, y_, state_)    
        y_new = np.argsort(pred, axis=1)[:, -beam_width]
        next_children = []
        for y_new_node, pred_node, things_new_node, state_new_node, node in zip(y_new, pred, things_new, state_new, children):
            y_nll_new_node = -np.log(pred_node[y_new_node])
            for y_new_node, y_nll_new_node in zip(y_new_node, y_nll_new_node):
                node_new = BeamTree(parent=node, state=state_new_node, value=y_new_node, cost=y_nll_new_node, things=things_new_node)
                next_children.append(node_new)

            next_children = sorted(next_children, key=lambda n:n.accumulated_cost)[:beam_width]

        K.sort(key=lambda n: n.accumulated_cost)
        return K[:top_k]

