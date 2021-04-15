using LinearAlgebra, Plots;gr();default(size=(300,200))

# Define the dataset Y
Nx = 400
x = range(-10,10,length=Nx) #  spatial coordinate (dx = 0.05)

Nt = 200
t = range(0,4π,length=Nt) # time coordinate (dt = 0.063)
dt=t[2]-t[1]

#  Building the two dimensional continous function f(x,t)
f_1(x,t)=sech(x+3)*exp(im*2.3*t)
f_2(x,t)=2*sech(x)*tanh(x)*exp(im*2.8*t)
f(x,t)=f_1(x,t)+f_2(x,t)

anim=Animation()
anim = @animate for tt in t
    plot( x, real(f_1.(x,tt)),title="f_1 for two seconds",xlim=(-10,10),ylim=(-1.,1.),leg=false,xlabel="x",ylabel="f_1(x,t)")
end
gif(anim, "img/DMD2_f_1_fps15.gif", fps = 15)

anim=Animation()
anim = @animate for tt in t
    plot( x, real(f_2.(x,tt)),xlim=(-10,10),ylim=(-1,1),title="f_2 for two seconds",leg=false,xlabel="x",ylabel="f_2(x,t)")
end
gif(anim, "img/DMD2_f_2_fps15.gif", fps = 15)

anim=Animation()
anim = @animate for tt in t
    plot( x, real(f.(x,tt)),xlim=(-10,10),ylim=(-2,2),title="f for two seconds",leg=false,xlabel="x",ylabel="f(x,t)")
end
gif(anim, "img/DMD2_f_fps15.gif", fps = 15)

# 2D Contour Plot of the data matrix Y
contour(x,t,f,levels=20)
xlabel!("x")
ylabel!("t")
title!("Contour Plot")

# 3D Surface Plot of the dataset
#my_cg = cgrad([:blue, :yellow])
surface(x,t,real(f), xaxis="x", yaxis="t",zaxis="Y", title="Surface Plot")

#  Building the discrete spatio-temporal series Y 
#Build of the spatio-temporal x vs t data matrix Y
Y = Complex.(zeros(Nx,Nt))
j=1
for tt in t
    Y[:,j] = f.(x,tt)  
    j+=1                                           
end
println("\rThe Data Matrix Y = \r")
display(Y)

# another way to create the data matrix Y (like the matlab meshgrid function )
X = repeat(reshape(x, :, 1), 1,length(t))

T = repeat(reshape(t,1,:),  length(x),1)

YY = map(f, X, T)

# 2D Contour Plot of the data matrix Y
contour(X[:,1],T[1,:],real(YY)',levels=20)
ylabel!("t")
title!("Contour Plot")

surface(X[:,1],T[1,:],real(YY)',xaxis="x", yaxis="t",zaxis="Y", title="Surface Plot Y=f(x,t)")
savefig("Surface Plot Y=f(x,t).png")
#= Interactive Plot 
xlabel!("x")
using PlotlyJS
trace=PlotlyJS.surface(x=x,y=t,z=real(Y), xlabel="x",colorscale="Viridis")
layout=PlotlyJS.Layout(title="Interactive Plot")
PlotlyJS.plot(trace,layout) =#

# Data decomposition with the singular value decomposition 
# U matrix (spatial Correlations); Σ matrix (weighting of projections), Vt (time dynamics)
Y1 = Y[:,1:end - 1]

Y2 = Y[:,2:end]

U, Σ, V=svd(Y1)

#= plot of the singular values (see the  two most dominants features due to the two ranks dataset 
(the two bumps with their temporal dynamics) The very good decay shows the existence of a low rank subspace =#
plot(Σ, yaxis=:log, xlim=(0,10),shape =:circle)

# Plot of the POD modes (Uj ; j={1,2}). Due to the very differents spatial scales, the POD modes extracted have a physical meanng 
plot(x,real(U[:,1]))
plot!(x,real(U[:,2]))
xlabel!("x")
ylabel!("u1(x), u2(x)")
title!("POD modes") # Spatial modes 

#plot of the Mode coefficents (σjVtj ; j={1,2]})
plot(0:0.01:1.98,real(Σ[1]*V[:,1]))
plot!(0:0.01:1.98,real(Σ[2]*V[:,2]))
xlabel!("t");ylabel!("a1(t), a2(t)");title!("Mode coefficients") # Temporal modes

# SVD and rank-2 truncation
r = 5
Uᵣ = U[:,1:r]
Σᵣ = diagm(Σ[1:r])
Vᵣ = V[:,1:r];

# Build Ã and DMD Modes
Ã = Uᵣ' * Y2 * Vᵣ / Σᵣ
Λ, W = eigen(Ã, sortby=nothing)
Φ = Y2 * Vᵣ / Σᵣ * W

# DMD Spectra
Ω = log.(Λ)/dt

# DMD mode amplitude
y₁=Y1[:,1]
b=Φ\y₁

time_dynamics=Complex.(zeros(r,length(t)))

for i in 1:length(t)
    time_dynamics[:,i] =b.*exp.(Ω* t[i])
end    

y₂=Φ*Diagonal(Λ)*b
YDMD=Φ*time_dynamics



surface(x,t,real(YDMD'),xaxis="x", yaxis="t",zaxis="YDMD",title="Surface Plot YDMD")
#Animation of the DMD1 mode
Y_DMD1=real(Φ[:,1]*time_dynamics[1,:]')
anim=Animation()
j=1
anim = @animate for tt in t
    plot( x, Y_DMD1[:,j],xlim=(-10,10),ylim=(-1.,1.),leg=false,title="DMD1 mode",xlabel="x",ylabel="Y_DMD1")
    j+=1
end
gif(anim, "img/DMD2_Y_DMD1_fps15.gif", fps = 15)

#Animation of the 2nd DMD mode
Y_DMD2=real(Φ[:,2]*time_dynamics[2,:]')

anim=Animation()
j=1
anim = @animate for tt in t
    plot( x, Y_DMD2[:,j],xlim=(-10,10),ylim=(-1.,1.),leg=false,title="DMD2 mode",xlabel="x",ylabel="Y_DMD2")
    j+=1
end
gif(anim, "img/DMD2_Y_DMD2_fps15.gif", fps = 15)

#Animation of Yrecons = Y_DMD1+Y_DMD2
Yrecons=Y_DMD1+Y_DMD2
anim=Animation()
j=1
anim = @animate for tt in t
    plot( x, Yrecons[:,j],xlim=(-10,10),ylim=(-2.,2.),leg=false,title="Reconstruction of Y",xlabel="x",ylabel="Yrecons")
    j+=1
end
gif(anim, "img/DMD2_Yrecons_fps15.gif", fps = 15)



