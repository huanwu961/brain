# Brain
## Nueron Model
### 1. Parameters
* Global parameters (system level)
    * total number of nuerons $n$
    * the distribution of the axion output shift length $L$ ~ $Poisson(\mu)$
    * one neuron can connect to $m$ neurons nearby.
    * time $t$
* Local parameters (neuron level)
    * internal parameter
        * the index of nueron is $i$
        * one nueron has one internal state $x_i \in [0, 1]$
        * culmulated incoming weighted states $c_i$
        * culmulated incoming weight sum $s_i$
        * one neuron can connect to a block of neuron nearby, with center shift $l_i$

    * external parameter
        * If there is a connection from neuron $i$ to $j$, the weight is denoted by $w_{ij}\in[0, 1]$
        * neuron correlation coefficient $C_{ij}$, describes whether there is a positive or negetive relation between neurons, used to update weight


### 2. Update Rules
* propagate forward
    * for node $i$ and its neighor $j \in N_i$ , pass the current weight $w_{ij}$ and current status $x_i$ from $i$ to $j$.
    * node $j$ receives the parameter $w_{ij}$ and $x_i$ then adds $w_{ij}x_i$ to $c_j$ and $w_{ij}$ to $s_j$
    $$c^{t+1}_j = \sum_i w^t_{ij}x^t_i$$
    $$s^{t+1}_j = \sum_i w^t_{ij}$$
* calculate new neuron state
    * for each neuron $i$ the new state is the weighed sum of the incoming states.
    $$x^{t+1}_i = \frac{c^{t+1}_i}{s^{t+1}_i} = \frac{\sum_i w^t_{ij}x^t_i}{\sum_i w^t_{ij}}$$

* calculate new weight
    * New weight is calculated according to the causal relation, neuron states and previous weight.
        * For each edge $i \to j$, if the new neuron state $x^{t+1}_j$ and old neuron state $x^{t}_i$ both larger or smaller then 0.5, then $i \to j$ has positive causal relationship, otherwise negetive. the correlation parameter is given by $$C^{t+1}_{ij} = (1-2x^{t}_i)(1-2x^{t+1}_j)$$
        * If the both nuerons are both inactive, although they has positive relation, the relation should not considered to be strong. So we consider the product of states as an important factor. $$x^{t+1}_{ij} = x^{t}_ix^{t+1}_j$$
        * If the previous weight tends to be weak, then it should tends to insist itself, meaning the closer to 0 or 1, the weight change less. $$dw^{t+1}_{ij}=
            \begin{cases}
            w^{t}_{ij}& 0<x<0.5 \\
            1-w^{t}_{ij}& 0.5<x<1
            \end{cases}$$
    * after combine these three componets, we have our final update formular $$w^{t+1}_{ij} = w^{t}_{ij} + \Delta w^{t+1}_{ij}$$ with  $$\Delta w^{t+1}_{ij} = C^{t+1}_{ij} \cdot x^{t+1}_{ij} \cdot dw^{t+1}_{ij}$$

### Network Architacture
* Way of connetion
    * Neurons will be stored in several arrays with different parameters. The row index $i$ means param[i] belongs to neuron $i$
    * Then the neuron spacial distance is defined by the memory distance, i.e. row index distance $|i - j|$.
    * For neuron $i$, the connection area is $[i + l_i + 1, i + l_i + m]$ ($l$ can be negetive or positive)
* Small world network
    * High clustering and low average path length.
    * Optimal information transformation.
    * Two critical parameters:
        * axion output shift length $l_i$
        * max connections for each neuron $m$
    * small $l_i$ means the output shift is nearby, and most neurons have a low axion output shift, hence high clustering coefficient.
    * large $l_i$ means the output is far away spacially. This will effiently lower the average path length of the whole network.

### Implementation
* InputStream
    * Goal
        * Receive ayntronization input source
            * Image
            * Sound
        * Convert them into 
* Network
* OutputStream

