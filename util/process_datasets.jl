using MatrixMarket

@time M = MatrixMarket.mmread("./wikipedia_link_oc.mtx")

@time M = M + M'

@time MatrixMarket.mmwrite("./wikipedia_link_oc-symmetric.mtx", M)
