# Robust and Efficient Kernel Hyperparameter Paths with Guarantees

**auto finetuning, svm, parameters, kernel parameter, choose parameters **

The results of non-linear SVM classifications heavily rely on the choose of the kernel hyperparameter, i.e. , \lambda in $k(x,y)=\exp(-\lambda \norm{x-y}²)$ in the kernel function.
This repository contains an effective algorithmn that calculates an approximate entire solution path of the objective function with respect to the hyperparameter within the interval **[2^{-10},2^{10}]** without numerical issues by which the exact algorithms suffer.
More details can be found in the corresponding [paper][paper]. 

For the matrix calculation the [library Eigen][eigen] was used in combination with [OpenMP][openmp]. The backend solver is [libSVM][libsvm]

The program reads the problem description in the default libSVM format. See the documented source for more information or run the compiled program without any parameters to get help.


### weblinks

 * [jmlr](http://jmlr.org/proceedings/papers/v32/giesen14.html) entry
 * [paper](http://jmlr.org/proceedings/papers/v32/giesen14.pdf) pdf format

 
[jmlr]:http://jmlr.org/proceedings/papers/v32/giesen14.html
[paper]:http://jmlr.org/proceedings/papers/v32/giesen14.pdf
[eigen]:http://eigen.tuxfamily.org/index.php?title=Main_Page
[libsvm]:http://www.csie.ntu.edu.tw/~cjlin/libsvm/
[openmp]:http://openmp.org/wp/





