function trstiff(coo,E,A)
# Function that computes stiffness matrix for truss element
# (file name trstiff.m)
L=sqrt((coo[2,1]-coo[1,1])^2+(coo[2,2]-coo[1,2])^2)
C=(coo[2,1]-coo[1,1])/L
S=(coo[2,2]-coo[1,2])/L
# Stiffness matrix
STF=E*A/L*[C^2 S*C -C^2 -S*C;
S*C S^2 -S*C -S^2;
-C^2 -S*C C^2 S*C;
-S*C -S^2 S*C S^2];
return(STF)
end
# Main program for truss structure
# Coordinates - units L[mm]
function truss_str(E,A)
COO=[0 0; 2 0; 4 0 ;1 1; 3 1; -1 1; 5 1]
COO=1000*COO
DOF=size(COO,1)*2
# Elements
ELM=[1 2; 2 3; 6 4; 4 5; 5 7; 1 6; 1 4; 2 4; 2 5; 3 5; 3 7]
# Properties (units F[N], L[mm])
#E=125000
#A=6
# Assembling of stiffness matrix
MSTF=zeros(DOF,DOF)
X=zeros(2,2)
for i in 1:size(ELM,1)
X[1,:]=COO[ELM[i,1],:]
X[2,:]=COO[ELM[i,2],:]
STF=trstiff(X,E,A)
MSTF[2*ELM[i,1]-1:2*ELM[i,1],2*ELM[i,1]-
1:2*ELM[i,1]]=MSTF[2*ELM[i,1]-1:2*ELM[i,1],2*ELM[i,1]-
1:2*ELM[i,1]]+STF[1:2,1:2]
MSTF[2*ELM[i,1]-1:2*ELM[i,1],2*ELM[i,2]-
1:2*ELM[i,2]]=MSTF[2*ELM[i,1]-1:2*ELM[i,1],2*ELM[i,2]-
1:2*ELM[i,2]]+STF[3:4,1:2]
MSTF[2*ELM[i,2]-1:2*ELM[i,2],2*ELM[i,1]-
1:2*ELM[i,1]]=MSTF[2*ELM[i,2]-1:2*ELM[i,2],2*ELM[i,1]-
1:2*ELM[i,1]]+STF[1:2,3:4]
MSTF[2*ELM[i,2]-1:2*ELM[i,2],2*ELM[i,2]-
1:2*ELM[i,2]]=MSTF[2*ELM[i,2]-1:2*ELM[i,2],2*ELM[i,2]-
1:2*ELM[i,2]]+STF[3:4,3:4]
end
# Constrains
con=[12 13 14 15] # constrained DOF
CDOF=size(con,2)
# Forces
dF=zeros(DOF,1)
dF[2]=-5000
dF[6]=-2000
# Reduced stiffness matrix & force vector
RSTF=MSTF[1:DOF-CDOF,1:DOF-CDOF]
dFR=dF[1:DOF-CDOF]
# Solving for displacements
d=inv(RSTF)*dFR
# End of main program
return(d)
end
function doe()
    cnt = 1
    U=zeros(10)
    for E in [125000 150000 175000 200000]
        for A in [6 8 10 12]
            Disp = truss_str(E, A)
            #show(Disp)
            if cnt > 1
                U = [U Disp]
            else
                U = Disp
            end
            cnt += 1
        end
    end
    #show(U)
    return (U)
end
# Execute The design of experiments (DOE)
U=doe()
println("\nThe Data Matrix U coming out from the DOE\n")
display(U)
#
using LinearAlgebra
# Proper Orthogonal Decomposition of the Data Matrix U
#F=eigen(U*U',sortby= x -> -abs(x))
F=eigen(U*U')
println("\nEigenValues λ\n")
show(F.values)
println("\n\nEigenVectors Matrix Φ\n")
Φ=F.vectors
display(Φ)
println("\nAmplitude Matrix A\n")
A=F.vectors'*U
display(A)
println("\nReconstruction of U=ΦA\n")
Urecons=Φ*A
display(Urecons)
println("\nReduction of Uapprox=Φ10*A10'~U\n")
Uapprox=Φ[:,10]*A[10,:]'
display(Uapprox)

#Calculate the SVD
Z=svd(U)
println("\nSVD decomposition=\r")
display(Z)
println("\nUapprox based on the first energetic mode=\r")
Ũ=Z.U[:,1]*(diagm(Z.S)*Z.Vt)[1,:]'
println("\nŨ=\r")
display(Ũ)
