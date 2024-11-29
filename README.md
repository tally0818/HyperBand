# HyperBand

 Following is the implementation based on : [paper](http://arxiv.org/pdf/1603.06560v4)

  

 To develop more efficient search methods, the problem of hyperparameter optimization(HPO) has recently been dominated by _Bayesian optimization_ methods that focus on optimizing hyperparameter _configuration selection_. However, these methods tackle the fundamentally challenging problem of simultaneously fitting and optimizing a high-dimensional, non-convex function with unknown smoothness, and possibly noisy evaluations. 

  

 In this situation, our authors suggest _HyperBand_, an orthogonal approach to HPO that focuses on speeding up _configuration evaluation_ by allocating more resources(can take various forms, including the size of the training set, numbers of features/iterations, …) to promising hyperparameter configurations while quickly eliminating poor ones.

  

 The main idea of the Hyperband algorithm is trying _Successive Halving algorithms_ on different _“exploration vs. exploitation trade-offs”_ while the total budget of resources is fixed. According to the paper, testing 5 different trade-offs will be reasonable. The pseudo-code of HyperBand follows.

  

 By numerically analyzing the pseudo code, we can find out that $R_{tot} = R( s_{max} + 1)^{2}$ total resources are allocated when the single HyperBand(R,) is called, and this algorithm considers ntot\=s=0smaxceil(smax+1s+1s) numbers of different configurations and for last, the configuration with the maximum resource will be allocated total Rmax\=R\-\-smax\-1 resources. At this moment, we can find out that the maximum resource assigned to a single configuration contradicts with the definition of R. We can fix this problem by modifying the code to remember each configuration's trained state. By making this change, we can optimize the total resource allocated to a single call as Roptimized tot\=(1-1)R(smax+1)2+1R(smax+1)ln(smax) and maximum resource allocated to a single configuration as R. By optimizing like this, it is essentially the same as HyperBand, but the total resources during execution are reduced. This is expected to save time in cases where large-scale models or long learning times are required. However, for petite models or short training times, the benefits are expected to be offset by storage/load overhead.

  

 Now let's get back to our reference paper. To compare HyperBand’s performance with other techniques, our authors designed an experiment comparing HyperBand’s performance with 3 well-known Bayesian optimization algorithms - SMAC, TPE, Spearmint, random(random search), random\_2x(random search with the twice or total resources), bracket s= 4 (repeating the most exploratory bracket 4 times) while using iterations, data set subsamples, and feature samples as a resource. The performance was evaluated using LeNet on CIFAR-10, SVHN, and MRBI datasets. One of the results follows.

  

 As you can see from the results of previous experiments, we are exploring various “exploration vs exploitation trade-offs”. As a result, faster convergence is possible than Bayesian Optimization. Therefore, good performance can be achieved efficiently in environments where model tuning time is limited. However, in unrestricted environments, existing optimization techniques yield higher performance.

This is because there is no exchange of information obtained when Hyperband pulls (train\_and\_return\_loss) each arm (configuration). To solve this, approaches such as BOHB exist. In my opinion, the attempt to incorporate Contextual Bandit also looks good.

  

 In my code, I implemented SearchSpace, HyperBand, Optimized\_HyperBand, Bracket and RandomSearch. First, SearchSpace is the part where the user can sample each configuration by determining each hyperparameter's distribution. Second, HyperBand is the same Hyperband algorithm proposed in the paper. The Optimized\_HyperBand is the version that remembers each configuration's trained state. Next, Bracket is the modified Hyperband that tries the most explorative bracket several times. At last, RandomSearch is implemented based on R, ,  instead of n, the number of configurations RandomSearch will try. This is to make a Fair comparison with the HyperBand algorithm. When RandomSearch gets R, as a input, it computes n to try n configurations with the same total resource and same average resource per configuration with HyperBand(R,). An example of usage is at main.py.
