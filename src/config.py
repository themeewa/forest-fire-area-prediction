from scipy.stats import loguniform
alpha = 0.99
gamma = 0.1
result_prefix = "dropfinal_1"

kernelridge__alpha = loguniform(alpha, 2)
kernelridge__gamma = loguniform(gamma, 1)

result_path = "../results/KernelRidge_"
