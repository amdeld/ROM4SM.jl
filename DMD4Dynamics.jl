using LinearAlgebra, Plots;gr();default(size=(300, 200))

# Define the dataset Y
Nx = 401
x = range(-2, 2, length=Nx) #  spatial coordinate (dx = 0.01)

Nt = 201
t = range(0, 2, length=Nt) # time coordinate (dt = 0.01)
dt = t[2] - t[1]
A1 = 1.
x1 = 0.5
σ1 = 0.2
f1 = 1.

A2 = 2.
x2 = -1.
σ2 = 0.2
f2 = 4.

# gaussian processes
y1(x) = A1 * exp(-((x - x1)^2) / (2 * σ1^2))
y2(x) = A2 * exp(-((x - x2)^2) / (2 * σ2^2));

# plot of the two spatial bumps
plot(x,y1)
plot!(x,y2,xlabel="x",ylabel="y1,y2",ylim=(-0.001, 2.))

# plot of the two temporal dynamics
plot(t,cos.(2 * π * f1 * t))
plot!(t,cos.(2 * π * f2 * t),xlabel="t",ylabel="cos",xlim=(0, 1),ylim=(-1, 1))

#  Building the two dimensional continous function f(x,t)
f_1(x,t) = A1 * exp(-((x - x1)^2) / (2 * σ1^2)) * cos(2 * π * f1 * t)
f_2(x,t) = A2 * exp(-((x - x2)^2) / (2 * σ2^2)) * cos(2 * π * f2 * t)
f(x,t) = f_1(x, t) + f_2(x, t)

anim = Animation()
anim = @animate for tt in t
    plot(x, f_1.(x, tt), xlim=(-2, 2), title="f_1", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f_1(x,t)")
end
gif(anim, "img/DMD_f_1_fps15.gif", fps=15)

anim = Animation()
anim = @animate for tt in t
    plot(x, f_2.(x, tt), xlim=(-2, 2), title="f_2", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f_2(x,t)")
end
gif(anim, "img/DMD_f_2_fps15.gif", fps=15)

plot(x, f.(x, 0), xlim=(-2, 2), title="f at t=0. sec", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f(x,t)")

anim = Animation()
anim = @animate for tt in t
    plot(x, f.(x, tt), xlim=(-2, 2), title="f(x,t)=f1(x,t)+f2(x,t)", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f(x,t)")
end
gif(anim, "img/DMD_f_fps15.gif", fps=15)

# 2D Contour Plot of the data matrix Y
contour(x,t,f,levels=20)
xlabel!("x")
ylabel!("t")
title!("Contour Plot")

# 3D Surface Plot of the dataset
# my_cg = cgrad([:blue, :yellow])
surface(x,t,f, xaxis="x", yaxis="t",zaxis="Y", title="Surface Plot")

#  Building the discrete spatio-temporal series Y 
# Build of the spatio-temporal x vs t data matrix Y
Y = Complex.(zeros(Nx, Nt))
j = 1
for tt in t
    Y[:,j] = y1.(x) * cos(2 * π * f1 * tt) + y2.(x) * cos(2 * π * f2 * tt) # gaussian processes 
    j += 1                                           # with their temporal dynamics
end
println("\rThe Data Matrix Y = \r")
display(Y)

# another way to create the data matrix Y (like the matlab meshgrid function )
X = repeat(reshape(x, :, 1), 1, length(t))
T = repeat(reshape(t, 1, :),  length(x), 1)
YY = map(f, X, T)

# 2D Contour Plot of the data matrix Y
contour(X[:,1],T[1,:],YY',levels=20)
xlabel!("x")
ylabel!("t")
title!("Contour Plot")

surface(X[:,1],T[1,:],YY', xaxis="x", yaxis="t",zaxis="YY", title="Surface Plot")

# Data decomposition with the singular value decomposition 
# U matrix (spatial Correlations); Σ matrix (weighting of projections), Vt (time dynamics)
Y1 = Y[:,1:end - 1]
Y2 = Y[:,2:end]
U, Σ, V = svd(Y1);

#= plot of the singular values (see the  two most dominants features due to the two ranks dataset 
(the two bumps with their temporal dynamics) The very good decay shows the existence of a low rank subspace =#
plot(Σ, yaxis=:log, xlim=(0, 10),shape=:circle)

# SVD and rank-2 truncation
r = 200
Uᵣ = U[:,1:r]
Σᵣ = diagm(Σ[1:r])
Vᵣ = V[:,1:r]

# Build Ã and DMD Modes
Ã = Uᵣ' * Y2 * Vᵣ / Σᵣ
Λ, W = eigen(Ã, sortby=nothing)
Φ = Y2 * Vᵣ / Σᵣ * W

# DMD Spectra
Ω = log.(Λ) / dt

# DMD mode amplitude
y₁ = Y1[:,1]
b = Φ \ y₁

ỹ₁ = Σᵣ * Vᵣ[1,:]
b = W * diagm(Λ) \ ỹ₁

time_dynamics = Complex.(zeros(r, length(t)))

for i in 1:length(t)
    time_dynamics[:,i] = b .* exp.(Ω * t[i])
end  

YDMD = Φ * time_dynamics

surface(x,t,real(YDMD'))

# Animation of the first DMD mode
Y_DMD1 = real(Φ[:,1] * time_dynamics[1,:]')
anim = Animation()
j = 1
anim = @animate for tt in t
    plot(x, Y_DMD1[:,j], xlim=(-2, 2), ylim=(-1., 1.), leg=false, title="DMD1 mode", xlabel="x", ylabel="Y_DMD1")
    j += 1
end
gif(anim, "img/DMD_Y_DMD1_fps15.gif", fps=15)

# Animation of the 2nd DMD mode 
Y_DMD2 = real(Φ[:,2] * time_dynamics[2,:]')
anim = Animation()
j = 1
anim = @animate for tt in t
    plot(x, Y_DMD2[:,j], xlim=(-2., 2.), ylim=(-1., 1.), leg=false, title="DMD2 mode", xlabel="x", ylabel="Y_DMD2")
    j += 1
end
gif(anim, "img/DMD_Y_DMD2_fps15.gif", fps=15)

# Animation of Yrecons = Y_DMD1+Y_DMD2
Yrecons = Y_DMD1 + Y_DMD2
anim = Animation()
j = 1
anim = @animate for tt in t
    plot(x, Yrecons[:,j], xlim=(-10, 10), ylim=(-2., 2.), leg=false, title="Reconstruction of Y", xlabel="x", ylabel="Yrecons")
    j += 1
end
gif(anim, "img/DMD2_Yrecons_fps15.gif", fps=15)

plot(x,t,Yrecons,st=:surface,xaxis="x", yaxis="t",zaxis="Yrecons")
title!("Reconstruction first two modes")

A1 = 1.25
x1 = 0.25
σ1 = 0.4
f1 = 2.

A2 = 1.5
x2 = -0.25
σ2 = 0.2
f2 = 3.

# gaussian processes
yn1(x) = A1 * exp(-((x - x1).^2) / (2 * σ1^2))
yn2(x) = A2 * exp(-((x - x2).^2) / (2 * σ2^2));

plot(x,yn1)
plot!(x,yn2,xlabel="x",ylabel="yn1,yn2")

plot(t,sin.(2 * π * f1 * t))
plot!(t,sin.(2 * π * f2 * t),xlabel="t",ylabel="sin",xlim=(0, 1),ylim=(-1, 1))

#  Building the two dimensional continous function f(x,t)
fn_1(x,t) = A1 * exp(-((x - x1)^2) / (2 * σ1^2)) * sin(2 * π * f1 * t)
fn_2(x,t) = A2 * exp(-((x - x2)^2) / (2 * σ2^2)) * sin(2 * π * f2 * t)
fn(x,t) = f_1(x, t) + f_2(x, t)

anim = Animation()
anim = @animate for tt in range(0, 1, length=100)
    plot(x, fn_1.(x, tt), xlim=(-2, 2), title="fn_1 for one second", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f_1(x,t)")
end
gif(anim, "img/anim_fn_1_fps15.gif", fps=15)

anim = Animation()
anim = @animate for tt in range(0, 1, length=100)
    plot(x, fn_2.(x, tt), xlim=(-2, 2), title="fn_2 for one second", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f_2(x,t)")
end
gif(anim, "img/anim_fn_2_fps15.gif", fps=15)

anim = Animation()
anim = @animate for tt in range(0, 1, length=100)
    plot(x, f.(x, tt), xlim=(-2, 2), title="fn for one second", leg=false, ylim=(-2., 2.), xlabel="x", ylabel="f(x,t)")
end
gif(anim, "img/anim_fn_fps15.gif", fps=15)

YN = zeros(Nx, Nt)
j = 1
for tt in t
    YN[:,j] = yn1.(x) * sin(2 * π * f1 * tt) + yn2.(x) * sin(2 * π * f2 * tt) # gaussian processes 
    j += 1                                           # with their temporal dynamics
end

# Data decomposition with the singular value decomposition 
# U matrix (spatial Correlations); σ matrix (weighting of projections), Vt (time dynamics) 
Z = svd(YN)
#= plot of the singular values (see the  two most dominants features due to the two ranks dataset 
(the two bumps with their temporal dynamics). Again The very good decay shows the existence of a low rank subspace =#
plot(Z.S, yaxis=:log, xlim=(0, 10),shape=:circle)

#= Plot of the POD modes (Uj ; j={1,2})The POD modes are not so physical, 
a kind of blending of the bumps occurs due to the greedy nature of the SVD algorithm =#
plot(x,Z.U[:,1])
plot!(x,Z.U[:,2])
xlabel!("x")
ylabel!("u1(x), u2(x)")
title!("POD modes")

# plot of the Mode coefficents (σjVtj ; j={1,2]})
plot(t,Z.S[1] * Z.Vt[1,:],xlim=(0, 1))
plot!(t,Z.S[2] * Z.Vt[2,:],xlim=(0, 1))
xlabel!("Time");ylabel!("a1(t), a2(t)");title!("Mode coefficients")

# Animation of the POD1 mode over the first second
YN_POD1 = Z.U[:,1] * Z.S[1] * Z.Vt[1,:]'
anim = Animation()
j = 1
anim = @animate for tt in range(0.01, 1, length=100)
    plot(x, YN_POD1[:,j], leg=false, xlim=(-2, 2), ylim=(-2., 2.), title="POD1 mode", xlabel="x", ylabel="Y_POD1")
    j += 1
end
gif(anim, "img/anim_YN_POD1_fps15.gif", fps=15)

# Animation of the POD2 mode over the first second
YN_POD2 = Z.U[:,2] * Z.S[2] * Z.Vt[2,:]'
anim = Animation()
j = 1
anim = @animate for tt in range(0.01, 1, length=100)
    plot(x, YN_POD2[:,j], leg=false, xlim=(-2, 2), ylim=(-2., 2.), title="POD2 mode", xlabel="x", ylabel="Y_POD2")
    j += 1
end
gif(anim, "img/anim_YN_POD2_fps15.gif", fps=15)

# Animation of Yrecons = Y_POD1+YPOD2 over the firt second
YNrecons = YN_POD1 + YN_POD2
anim = Animation()
j = 1
anim = @animate for tt in range(0.01, 1, length=100)
    plot(x, YNrecons[:,j], xlim=(-2, 2), leg=false, ylim=(-2., 2.), title="Reconstruction of Y based \n on the first two modes", xlabel="Space x", ylabel="Yrecons")
    j += 1
end
gif(anim, "img/anim_YNrecons_fps15.gif", fps=15)


