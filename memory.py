from collections import deque
import numpy as np
import random
import torch


class Memory:
    """
    Class of the uniform experience replay memory.
    """

    def __init__(self, max_size):
        """
        Description
        -------------
        Constructor of class Memory.

        Attributes & Parameters
        -------------
        max_size : Int, the maximum size of the replay memory
        buffer   : collections.deque object of maximum length max_size, the container
                   representing the replay memory
        """
        self.buffer = deque(maxlen=max_size)

    def add(self, experience, priority):
        """
        Add a new experience with its associated priority.
        """
        # Crea un nuovo nodo per il sum tree
        if not Node.saturated:
            leaf = Node(max_size=self.max_size, value=priority)
            index = leaf.index
            
            # Memorizza experience e leaf separatamente, evitando di creare un array che li combina
            self.buffer[0, index] = experience
            self.buffer[1, index] = leaf
            
            # Aggiungi il nodo al tree e al heap
            self.tree.add_leaf(leaf)
            self.heap.insert(leaf)
        else:
            # Ottieni l'indice da riutilizzare in base alla strategia di sliding
            if self.sliding == "oldest":
                index = Node.count % self.max_size
                Node.count += 1
            elif self.sliding == "random":
                index = np.random.randint(0, self.max_size)
                
            # Recupera il nodo esistente
            leaf = self.buffer[1, index]
            
            # Aggiorna il buffer senza creare un array composito
            self.buffer[0, index] = experience
            # Il nodo leaf rimane lo stesso, aggiorniamo solo la sua priorità
            self.update(index, priority)

    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" experiences from the replay buffer.

        Parameters
        -------------
        batch_size : Int, the number of experiences to sample.

        Returns
        -------------
        List containing the sampled experiences.
        """
        return random.sample(self.buffer, batch_size)


class Node:
    # This counter is a class attribute and will define indices of leaves in the array
    # where we store them.
    count = 0
    # Storing the index will help us deal with the reaplay buffer when it gets saturated.
    saturated = False  # This boolean will turn True when Node.count >= max_size

    def __init__(
        self,
        max_size,
        index_heap=None,
        l_child=None,
        r_child=None,
        children_heap=[],
        parent=None,
        parent_heap=None,
        value=0.0,
        sliding="oldest",
    ):
        """
        Description
        ------------------------
        Constructor of class Node, this class represents objects describing how nodes are
        interlinked in both a sum tree and a heap.

        Parameters & Attributes
        ------------------------
        max_size      : Int, maximum size of the replay buffer where we store the leaves.
        index         : Int, index of the leaf on the storing array (only useful when the
                        node is actually a leaf).
        index_heap    : Int, index of the leaf on the storing array of the heap.
        l_child       : None or object of class Node, Left Child in the sum tree.
        r_child       : None or object of class Node, Right Child in the sum tree.
        children_heap : Sorted List containing the children of the node in the heap.
                        Sorting makes method sift_down of Heap easy.
        parent        : None or object of class Node, parent in sum tree. Setting the
                        parent helps us updating the value of each node
                        starting from a changed leaf up to the root.
        parent_heap   : None or object of class Node, parent in the heap.
        value         : Float, sum over all values of the subtree spanned by this node as
                        a root (TD error magnitude in case of a leaf).
        sliding       : String in ['oldest', 'random'], when the tree gets saturated and a
                        new experience comes up.
                           - 'oldest' : Oldest leaves are the first to be changed.
                           - 'random' : Random leaves are changed.
        leaf          : Boolean, whether the node is a leaf in the sum tree or not (True
                        when both l_child and r_child are None).
        leaf_heap     : Boolean, whether the node is a leaf in the heap or not (True when
                        both l_child_heap and r_child_heap are None).
        level         : Int, it specifies the hierarchy of nodes in the sum tree starting
                        from the leaves (0) and up to the root.
        complete      : Boolean, whether the node has both of its children in the same
                        level or not.
        """
        self.max_size = max_size
        self.index = Node.count
        self.index_heap = index_heap
        self.l_child = l_child
        self.r_child = r_child
        self.children_heap = sorted(children_heap, reverse=True)
        self.parent = parent
        self.parent_heap = parent_heap
        self.value = value
        self.leaf = (l_child is None) & (r_child is None)
        self.leaf_heap = len(children_heap) == 0
        self.complete = False
        if self.leaf:
            # Set the leaf index to class attribute count.
            self.index = Node.count
            # Increment class attribute count to account for tree saturation.
            Node.count += 1
            self.level = 0  # Level 0 because it is a leaf.
            # Update class attribute count (tree saturation status).
            Node.saturated = Node.count >= self.max_size

        elif self.r_child is None:
            # Every node that is not a leaf has at least a left child, in case it does not
            # have a right child, the node's level is the increment by 1 of the level of
            # its left child.
            self.level = self.l_child.level + 1

        else:
            # In case the node has both children, it takes the increment by 1 of the
            # minimum level. The reason is that when the tree evolves
            # by adding new leaves, this node will eventually have its children change
            # until reaching the mentioned minimum level.
            self.level = min(self.l_child.level, self.r_child.level) + 1
            self.complete = self.l_child.level == self.r_child.level

    def reset_count():
        """
        Description
        ------------------------
        Class method, resets class attribute count to 0

        Returns
        ------------------------
        """
        Node.count = 0
        Node.saturated = False

    def update_complete(self):
        """
        Description
        ------------------------
        Update the status (complete or not) of the current node, this can be triggered by
        an update of its children.
        """
        assert not self.leaf, "Do not update the status of a leaf"
        if self.r_child is None:
            pass
        else:
            self.complete = self.l_child.level == self.r_child.level

    def update_level(self):
        """
        Description
        ------------------------
        Update the level of the current node, this can be triggered by an update of its
        children.
        """
        if self.r_child is None:
            self.level = self.l_child.level + 1
        else:
            self.level = min(self.l_child.level, self.r_child.level) + 1

    def update_value(self):
        """
        Description
        ------------------------
        Update the value of the node after setting its left and right children.
        """
        self.value = self.l_child.value + self.r_child.value

    def update(self):
        """
        Description
        ------------------------
        Update level, status and value attributes of the node.
        """
        self.update_level()
        self.update_complete()
        self.update_value()

    def update_leaf_heap(self):
        """
        Description
        ------------------------
        Update the attribute leaf_heap.
        """
        self.leaf_heap = len(self.children_heap) == 0

    def set_l_child(self, l_child):
        """
        Description
        ------------------------
        Set the left child of the node.
        """
        self.l_child = l_child

    def set_r_child(self, r_child):
        """
        Description
        ------------------------
        Set the right child of the node.
        """
        self.r_child = r_child

    def set_children_heap(self, children_heap):
        """
        Description
        ------------------------
        Set the nodes' children in the heap.
        """
        self.children_heap = children_heap
        self.children_heap.sort(reverse=True)
        for child in children_heap:
            child.set_parent_heap(self)

    def replace_child_heap(self, child_origin, child_new):
        """
        Description
        ------------------------
        Replace a child among the children of the node in the heap.
        """
        assert child_origin in self.children_heap, (
            "The child you want to replace does not belong to "
            "the children of current node!"
        )
        for i, child in enumerate(self.children_heap):
            if child == child_origin:
                self.children_heap[i] = child_new
        self.children_heap.sort(reverse=True)
        child_new.set_parent_heap(self)

    def add_child_heap(self, child):
        """
        Description
        ------------------------
        Add a new child in the heap to the current node when it does not already have two
        children.
        """
        assert len(self.children_heap) < 2, (
            "The node already has 2 children, cannot add a child; consider replacing."
        )
        self.children_heap.append(child)
        self.children_heap.sort(reverse=True)
        child.set_parent_heap(self)

    def set_parent_heap(self, parent_heap):
        """
        Description
        ------------------------
        Set the parent of the node in the heap.
        """
        self.parent_heap = parent_heap

    def set_index_heap(self, index_heap):
        """
        Description
        ------------------------
        Set the index of the node in the heap.
        """
        self.index_heap = index_heap

    def __lt__(self, node):
        """
        Description
        ------------------------
        Less-than method, useful for sorting nodes in the heap.
        The node with higher value (priority) is considered "greater".
        """
        return self.value < node.value


def retrieve_leaf(node, s):
    """
    Description
    ------------------------
    Retrieve the index of a leaf given a random number s in [0, node.value].
    """
    if node.leaf:
        return node.index
    elif node.l_child.value >= s:
        return retrieve_leaf(node.l_child, s)
    else:
        return retrieve_leaf(node.r_child, s - node.l_child.value)


# Vectorized retrieve_leaf
retrieve_leaf_vec = np.vectorize(retrieve_leaf, excluded=set([0]))


def retrieve_value(node):
    """
    Description
    ------------------------
    Retrieve the value of a node (intended for vectorized use).
    """
    return node.value


# Vectorized retrieve_value
retrieve_value_vec = np.vectorize(retrieve_value)


class Heap:
    def __init__(self):
        """
        Description
        ------------------------
        Constructor of class Heap.

        Attributes:
          - track: list representing the heap.
          - root: the root node.
        """
        self.track = []
        self.root = None
        self.last_child = None

    def swap(self, child, parent):
        """
        Description
        ------------------------
        Swap the parent-child relationship between two nodes.
        """
        child_children_heap, parent_children_heap, grand_parent = (
            child.children_heap,
            parent.children_heap,
            parent.parent_heap,
        )
        child_index_heap, parent_index_heap = child.index_heap, parent.index_heap
        child.set_index_heap(parent_index_heap)
        parent.set_index_heap(child_index_heap)
        parent.set_children_heap(child_children_heap)
        child.set_children_heap(parent_children_heap)
        child.replace_child_heap(child, parent)
        if grand_parent is not None:
            grand_parent.replace_child_heap(parent, child)
        else:
            child.set_parent_heap(None)
            self.root = child
        self.track[child.index_heap] = child
        self.track[parent.index_heap] = parent

    def sift_up(self, node):
        """
        Description
        ------------------------
        Update the heap when a node's value increases.
        """
        parent = node.parent_heap
        changed = False
        while (parent is not None) and (node > parent):
            self.swap(node, parent)
            parent = node.parent_heap
            changed = True
        return changed

    def sift_down(self, node):
        """
        Description
        ------------------------
        Update the heap when a node's value decreases.
        """
        children = node.children_heap
        changed = False
        while (len(children) != 0) and (children[0] > node):
            self.swap(children[0], node)
            children = node.children_heap
            changed = True
        return changed

    def update(self, node, value):
        """
        Description
        ------------------------
        Update the node's value and propagate the change in the heap.
        """
        value_prev = node.value
        node.value = value
        if value < value_prev:
            self.sift_down(node)
        else:
            self.sift_up(node)

    def insert(self, node):
        """
        Description
        ------------------------
        Insert a new node into the heap.
        """
        self.track.append(node)
        node.set_index_heap(len(self.track) - 1)
        if self.root is None:
            self.root = node
        else:
            parent = self.track[(node.index_heap - 1) // 2]
            parent.add_child_heap(node)


class SumTree:
    def __init__(self, max_size):
        """
        Description
        ------------------------
        Constructor of class SumTree.

        Parameters:
          - max_size: maximum number of leaves.
        """
        self.max_size = max_size
        self.sub_left = None
        self.parents = deque()
        self.children = deque()
        self.complete = False

    def add_leaf(self, node):
        """
        Description
        ------------------------
        Add a new leaf to the tree.
        """
        if self.sub_left is None:
            self.sub_left = node
            self.complete = True
        else:
            root = Node(self.max_size, l_child=self.sub_left)
            self.sub_left.parent = root
            self.parents.appendleft(root)
            self.children.append(node)
            self.complete = False
            if len(self.parents) >= 2:
                self.parents[-1].l_child = self.children[-2]
                self.children[-2].parent = self.parents[-1]
                self.parents[-1].r_child = self.children[-1]
                self.children[-1].parent = self.parents[-1]
                self.parents[-1].update()
                while self.parents[-1].complete:
                    node = self.parents.pop()
                    self.children.pop()
                    self.children[-1] = node
                    if len(self.parents) == 1:
                        break
                    self.parents[-1].l_child = self.children[-2]
                    self.children[-2].parent = self.parents[-1]
                    self.parents[-1].r_child = self.children[-1]
                    self.children[-1].parent = self.parents[-1]
                    self.parents[-1].update()
                if len(self.parents) >= 2:
                    for i in range(-2, -len(self.parents), -1):
                        self.parents[i].l_child = self.children[i - 1]
                        self.children[i - 1].parent = self.parents[i]
                        self.parents[i].r_child = self.parents[i + 1]
                        self.parents[i + 1].parent = self.parents[i]
                        self.parents[i].update()
                    self.parents[0].r_child = self.parents[1]
                    self.parents[0].update()
                else:
                    self.parents[0].r_child = self.children[0]
                    self.children[0].parent = self.parents[0]
                    self.parents[0].update()
                    if self.parents[0].complete:
                        root = self.parents.pop()
                        self.children.pop()
                        self.sub_left = root
                        self.complete = True
            elif len(self.parents) == 1:
                self.parents[0].r_child = self.children[0]
                self.children[0].parent = self.parents[0]
                self.parents[0].update()
                if self.parents[0].complete:
                    root = self.parents.pop()
                    self.children.pop()
                    self.sub_left = root
                    self.complete = True

    def sample_batch(self, batch_size=64):
        """
        Description
        ------------------------
        Sample a batch of leaf indices according to the sum tree distribution.
        """
        root = self.sub_left if (len(self.parents) == 0) else self.parents[0]
        ss = np.random.uniform(0, root.value, batch_size)
        return retrieve_leaf_vec(root, ss)

    def update(self, node):
        """
        Description
        ------------------------
        Update the tree by propagating the node's new value up to the root.
        """
        parent = node.parent
        parent.update_value()
        parent = parent.parent
        while parent is not None:
            parent.update_value()
            parent = parent.parent

    def retrieve_root(self):
        """
        Description
        ------------------------
        Retrieve the root node of the tree.
        """
        return self.sub_left if len(self.parents) == 0 else self.parents[0]


def retrieve_first(couple):
    return couple[0]


retrieve_first_vec = np.vectorize(retrieve_first)


class PrioritizedMemory:
    """
    Class of the prioritized experience replay memory.
    """

    def __init__(self, max_size, sliding="oldest"):
        """
        Constructor of class PrioritizedMemory.

        Parameters & Attributes
        ------------------------
        max_size : Int, maximum size of the replay memory.
        sliding : String in ['oldest', 'random'], strategy for replacing experiences when full.
        buffer : List of two lists, where:
                - buffer[0] stores experiences.
                - buffer[1] stores corresponding Node objects.
        tree : SumTree object for sampling.
        heap : Heap object for managing priorities.
        """
        self.max_size = max_size
        assert sliding in ["oldest", "random"], "sliding parameter must be either 'oldest' or 'random'"
        self.sliding = sliding
        
        # Initialize buffer as two lists
        self.buffer = [[], []]  # First list for experiences, second for leaves
        for _ in range(max_size):
            self.buffer[0].append(None)
            self.buffer[1].append(None)
        
        self.tree = SumTree(max_size=max_size)
        self.heap = Heap()

    def update(self, index, priority):
        """
        Update the priority of the experience at the given index.
        """
        node = self.buffer[1][index]
        self.heap.update(node, priority)
        self.tree.update(node)

    def add(self, experience, priority):
        """
        Add a new experience with its associated priority.
        """
        if not Node.saturated:
            leaf = Node(max_size=self.max_size, value=priority)
            index = leaf.index
            
            # Store experience and leaf separately
            self.buffer[0][index] = experience
            self.buffer[1][index] = leaf
            
            # Add the node to the tree and heap
            self.tree.add_leaf(leaf)
            self.heap.insert(leaf)
        else:
            # Get the index to reuse based on the sliding strategy
            if self.sliding == "oldest":
                index = Node.count % self.max_size
                Node.count += 1
            elif self.sliding == "random":
                index = np.random.randint(0, self.max_size)
                
            # Retrieve the existing node
            leaf = self.buffer[1][index]
            
            # Update the buffer without creating a composite array
            self.buffer[0][index] = experience
            # The leaf node remains the same, just update its priority
            self.update(index, priority)

    def sample(self, batch_size):
        """
        Randomly sample batch_size experiences from the replay buffer.
        Returns:
            - A list of experiences.
            - An array of corresponding indices.
        """
        indices = self.tree.sample_batch(batch_size)
        experiences = [self.buffer[0][i] for i in indices]
        return experiences, indices
    
    def highest_priority(self):
        """
        Return the highest priority in the replay buffer.
        """
        return self.heap.root.value

    def n_experiences(self):
        """
        Return the number of experiences stored so far.
        """
        return len(self.heap.track)

    def sum_priorities(self):
        """
        Return the sum of all priorities.
        """
        root = self.tree.retrieve_root()
        return root.value

    def retrieve_priorities(self, indices):
        """
        Return the priorities for the experiences at the given indices.
        """
        # Get the leaf nodes at the given indices
        leaves = [self.buffer[1][i] for i in indices]
        # Get the values/priorities from these leaves
        return retrieve_value_vec(np.array(leaves, dtype=object))