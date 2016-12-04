#=
NOT INTENDED TO BE RUN AS A SCRIPT
=#
using GraphGLRM, LowRankModels, DataFrames, DataArrays

wdi = readtable("cleaned.csv")
edges = Tuple{Int64,Int64}[]
for i in 1:(nrow(wdi) - 1)
  if (wdi[:Country_Code][i] == wdi[:Country_Code][i+1]) && (wdi[:Year][i] + 1 == wdi[:Year][i+1])
    push!(edges, (i,i+1))
  end
end

xgraph = IndexGraph([1:nrow(wdi)...], edges)

"""
Converts DataFrame of floating-point numbers to a DataMatrix
"""
function dataMatrix(d::DataFrame)
  res = @data(zeros(size(d)))
  for j in 1:length(d)
    for i in 1:nrow(d)
      res[i,j] = d[i,j]
    end
  end
  res
end

isntna(A) = !isna(A)

#The raw DataMatrix from this DataFrame
wdiA = dataMatrix(wdi[:,3:end])
obsgini = wdi[:SI_POV_GINI] |> isntna
findfirst(names(wdi) .== :SI_POV_GINI) - 2

##############################Validation Set##################################
wdiDfVal = wdi[obsgini,:]

#Use the subset of the data where the gini index is defined
valedges = Tuple{Int64, Int64}[]
#Data is sorted by country first then by year.
for i in 1:(nrow(wdiDfVal) - 1)
  if (wdiDfVal[:Country_Code][i] == wdiDfVal[:Country_Code][i+1]) && (wdiDfVal[:Year][i] + 1 == wdiDfVal[:Year][i+1])
    push!(valedges, (i, i+1))
  end
end

xgraphval = IndexGraph([1:nrow(wdiDfVal)...], valedges)

wdiValidation = wdiA[obsgini, :]
wdival, m, s = standardize(wdiValidation)

#Knock out 1/2 of non-NA values at random because this is number knocked out of original
#This knocks out about 40% of the gini coefficient values, about the same as are missing originally.
knockout = deepcopy(wdival)
knockout[rand(1:length(knockout), length(knockout)÷2)] = NA
sum(isna(knockout[:,12])) #474
kidxs = isna(knockout[:,12])
tidxs = !kidxs
ginicol = 12 #This is the column index for the gini index, the desired variable

ggstrain = GGLRM(wdival, HuberLoss(), QuadReg(1.), QuadReg(1.), 20, obs=observations(wdival))

#Use standardized data for smaller objective values and to allow proper graph regularization
#Huber loss for robustness, two quadratic regularizers to reduce overfitting
#First try 20 dimensions
ggs20 = GGLRM(knockout, HuberLoss(), QuadReg(1.), QuadReg(1.), 20, obs=observations(knockout))
fit!(ggs20) #709 objective value
whole_objective(ggstrain, ggs20.X*ggs20.Y)

(((ggs20.X*ggs20.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12])) |> abs2 |> mean
(((ggs20.X*ggs20.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12])) |> abs2 |> mean

#Next try 10
ggs10 = GGLRM(knockout, HuberLoss(), QuadReg(1.), QuadReg(1.), 10, obs=observations(knockout))
fit!(ggs10) #657 objective value
whole_objective(ggstrain, ggs10.X*ggs10.Y)

(((ggs10.X*ggs10.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12])) |> abs2 |> mean
(((ggs10.X*ggs10.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12])) |> abs2 |> mean

#Next try 5
ggs5 = GGLRM(knockout, HuberLoss(), QuadReg(1.), QuadReg(1.), 5, obs=observations(knockout))
fit!(ggs5) #808 objective value
whole_objective(ggstrain, ggs5.X*ggs5.Y)

(((ggs5.X*ggs5.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12])) |> abs2 |> mean
(((ggs5.X*ggs5.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12])) |> abs2 |> mean

#Next try 2
ggs2 = GGLRM(knockout, HuberLoss(), QuadReg(1.), QuadReg(1.), 2, obs=observations(knockout))
fit!(ggs2) #Hits a much higher objective value than the others
whole_objective(ggstrain, ggs2.X*ggs2.Y)

(((ggs2.X*ggs2.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12])) |> abs2 |> mean
(((ggs2.X*ggs2.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12])) |> abs2 |> mean

#Mean l1 relative error
(((ggs10.X*ggs10.Y)[:,12] - wdival[:,12])./(wdival[:,12])) |> abs |> mean
#Mean l2 relative error
(((ggs10.X*ggs10.Y)[:,12] - wdival[:,12])./(wdival[:,12])) |> abs2 |> mean
#Mean l1 relative error over knocked out values
((ggs10.X*ggs10.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
#Mean l2 relative error over knocked out values
((ggs10.X*ggs10.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean

#Mean squared error
((ggs10.X*ggs10.Y)[kidxs,12] - wdival[kidxs,12]) |> abs2 |> mean
#Median absolute error
((ggs10.X*ggs10.Y)[kidxs,12] - wdival[kidxs,12]) |> abs |> median

#mean absolute value
median(abs(wdival[:,12]))
#mean squared value
mean(abs2(wdival[:,12]))
#mean absolute value of knockouts
median(abs(wdival[kidxs,12]))
#mean squared value of knockouts
mean(abs2(wdival[kidxs,12]))

#########################GRAPH REGULARIZATION#################################
#Magnitude 0 graph regularization
graph_0 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 0., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_0, ProxGradParams(max_iter=1000))
whole_objective(graph_0, graph_0.X*graph_0.Y)
whole_objective(ggstrain, graph_0.X*graph_0.Y)

((graph_0.X*graph_0.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_0.X*graph_0.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_0.X*graph_0.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#Magnitude 1 graph regularization
graph_1 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 1., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_1, ProxGradParams(max_iter=1000))
whole_objective(graph_1, graph_1.X*graph_1.Y)
whole_objective(ggstrain, graph_1.X*graph_1.Y)

((graph_1.X*graph_1.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_1.X*graph_1.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_1.X*graph_1.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#Magnitude 2 graph regularization
graph_2 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 2., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_2, ProxGradParams(max_iter=1000))
whole_objective(graph_2, graph_2.X*graph_2.Y)
whole_objective(ggstrain, graph_2.X*graph_2.Y)

((graph_2.X*graph_2.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_2.X*graph_2.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_2.X*graph_2.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#Magnitude 3 graph regularization
graph_3 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 3., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_3, ProxGradParams(max_iter=1000))
whole_objective(graph_3, graph_3.X*graph_3.Y)
whole_objective(ggstrain, graph_3.X*graph_3.Y)

((graph_3.X*graph_3.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_3.X*graph_3.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_3.X*graph_3.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#Magnitude 4 graph regularization
graph_4 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 4., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_4, ProxGradParams(max_iter=1000))
whole_objective(graph_4, graph_4.X*graph_4.Y)
whole_objective(ggstrain, graph_4.X*graph_4.Y)

((graph_4.X*graph_4.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_4.X*graph_4.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_4.X*graph_4.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#Magnitude 5 graph regularization
graph_5 = GGLRM(knockout, HuberLoss(), GraphQuadReg(xgraphval, 5., 0.1), QuadReg(1.), 10, obs=observations(knockout))
fit!(graph_5, ProxGradParams(max_iter=1000))
whole_objective(graph_5, graph_5.X*graph_5.Y)
whole_objective(ggstrain, graph_5.X*graph_5.Y)

((graph_5.X*graph_5.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs |> mean
((graph_5.X*graph_5.Y)[kidxs,12] - wdival[kidxs,12])./(wdival[kidxs,12]) |> abs2 |> mean
((graph_5.X*graph_5.Y)[tidxs,12] - wdival[tidxs,12])./(wdival[tidxs,12]) |> abs2 |> mean

#writedlm("validationmatrix.txt", wdival)
#writedlm("validationerror.txt", wdival - graph_3.X*graph_3.Y)
testset1, testset2 = deepcopy(wdival), deepcopy(wdival)
testset1[rand(1:length(testset1), length(testset1)÷2)] = NA
testset2[rand(1:length(testset2), length(testset2)÷2)] = NA
testidx1 = isna(testset1[:,12])
testidx2 = isna(testset2[:,12])
tridx1 = !testidx1
tridx2 = !testidx2

graph_test1 = GGLRM(testset1, HuberLoss(), GraphQuadReg(xgraphval, 3., 0.1), QuadReg(1.), 10, obs=observations(testset1))
graph_test2 = GGLRM(testset2, HuberLoss(), GraphQuadReg(xgraphval, 3., 0.1), QuadReg(1.), 10, obs=observations(testset2))
fit!(graph_test1, ProxGradParams(max_iter=1000))
fit!(graph_test2, ProxGradParams(max_iter=1000))

graph_base1 = GGLRM(testset1, HuberLoss(), QuadReg(1.), QuadReg(1.), 10, obs=observations(testset1))
graph_base2 = GGLRM(testset2, HuberLoss(), QuadReg(1.), QuadReg(1.), 10, obs=observations(testset2))
fit!(graph_base1, ProxGradParams(max_iter=1000))
fit!(graph_base2, ProxGradParams(max_iter=1000))

((graph_test1.X*graph_test1.Y)[testidx1,12] - wdival[testidx1,12])./(wdival[testidx1,12]) |> abs2 |> mean
((graph_test1.X*graph_test1.Y)[tridx1,12] - wdival[tridx1,12])./(wdival[tridx1,12]) |> abs2 |> mean

((graph_test2.X*graph_test2.Y)[testidx2,12] - wdival[testidx2,12])./(wdival[testidx2,12]) |> abs2 |> mean
((graph_test2.X*graph_test2.Y)[tridx2,12] - wdival[tridx2,12])./(wdival[tridx2,12]) |> abs2 |> mean

((graph_base1.X*graph_base1.Y)[testidx1,12] - wdival[testidx1,12])./(wdival[testidx1,12]) |> abs2 |> mean
((graph_base1.X*graph_base1.Y)[tridx1,12] - wdival[tridx1,12])./(wdival[tridx1,12]) |> abs2 |> mean

((graph_base2.X*graph_base2.Y)[testidx2,12] - wdival[testidx2,12])./(wdival[testidx2,12]) |> abs2 |> mean
((graph_base2.X*graph_base2.Y)[tridx2,12] - wdival[tridx2,12])./(wdival[tridx2,12]) |> abs2 |> mean
