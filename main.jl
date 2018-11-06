using Plots
gr(size=(600,400))
default(fmt = :png)

function main()
    D = readcsv("dados.csv") 
    x = D[:,1]
    y = D[:,2]

    kfold(x, y)

    p = 8 
    xlin = linspace(extrema(x)..., 100)
    β = regressao_polinomial(x, y, p)
    ylin = β[1] * ones(100)
    for j = 1:p
        ylin .+= β[j+1] * xlin.^j
    end
    scatter(x, y, ms=3, c=:blue)
    plot!(xlin, ylin, c=:red, lw=2)
    png("ajuste")
    
    #y_pred = zeros()
    #for j = 1:p
        #y_pred .+= β[j+1] * x.^j
    #end
    #y_med = mean(y)
    #R2 = 1 - norm(y_pred - y)^2 / norm(y_med - y)^2  
end

function regressao_polinomial(x, y, p)
    m = length(x)
    A = [ones(m) [x[i]^j for i = 1:m, j = 1:p]]
    β = (A' * A) \ (A' * y)
    return β
end

function kfold(x, y; num_folds = 5, max_p=15)
    m = length(x)
    k = div(m, num_folds)
    I = randperm(m)
    x, y = x[I], y[I]
    E_treino = zeros(num_folds,max_p)
    E_teste = zeros(num_folds,max_p)
    for fold = 1:num_folds
        cjto_teste = k*(fold-1)+1:k*fold
        cjto_treino = setdiff(1:m, cjto_teste)
        x_tr, y_tr = x[cjto_treino], y[cjto_treino]
        for p = 1:max_p
            β = regressao_polinomial(x_tr, y_tr, p)
            y_pred = β[1] + sum(β[j+1]*x.^j for j=1:p)
            erro_treino = (1/(2 * lenght(cjto_treino))) * sum(y[i]-y_pred[i] for i=cjto_treino)^2
            erro_teste = (1/(2 * lenght(cjto_teste))) * sum(y[i]-y_pred[i] for i=cjto_teste)^2
            E_treino[fold,p] = erro_treino
            E_teste[fold,p] = erro_teste
        end
        #Em_teste = mean(E_teste[fold,:])
        #Em_treino = mean(E_treino[fold,:])
        plot!(1:max_p, E_treino[fold,:], c=:blue, ms=1, leg=false)
        plot!(1:max_p, E_teste[fold,:], c=:green, ms=1, leg=false)
        #plot!(1:max_p, Em_treino[fold,:], c=:blue, ms=3, leg=false)
        #plot!(1:max_p, Em_teste[fold,:], c=:green, ms=3, leg=false)
    end
    png("kfold")
end

main()
