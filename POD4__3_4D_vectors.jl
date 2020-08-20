using LinearAlgebra
#Consider a set of four three dimensional vectors that are
#collected in a matrix (3x4) U=[u1 u2 u3 u4]
U=[1 5 3 3;1 4 4 3;1 5 5 4]
println("\nU data matrix=\r")
display(U)
#Calculate The Covariance matrix C (3x3)
C=U*U'
println("\nC covariance matrix (U*U')=\r")
display(C)
#Calculate the eigenvalues and the eigenvectors
#F=eigen(C, sortby=x->abs(x))
F=eigen(C)
λ=F.values
println("\nλ eigenvalues=\r")
display(λ')
Φ=F.vectors
println("\nΦ eigenvectors matrix=\r")
display(Φ)
#Calculate the Amplitude matrix A = Φ'*U (3x4)
A=Φ'*U
println("\nA amplitude matrix=\r")
display(A)
#Check that U=Φ*A
Urec=Φ*A
println("\nUrec=Φ*A\r")
display(Urec)
#Keep the last direction and Amplitude Ũ=Φ3*A3
println("\nŨ3=\r")
Ũ3=Φ[:,3]*A[3,:]'
display(Ũ3)
u1=A[3,:][1]*Φ[:,3]
u2=A[3,:][2]*Φ[:,3]
u3=A[3,:][3]*Φ[:,3]
u4=A[3,:][4]*Φ[:,3]
#Keep the two last directions and Amplitude Ũ=Φ23*A23
println("\nŨ23=\r")
Ũ23=Φ[:,2:3]*A[2:3,:]
display(Ũ23)

#Calculte the SVD
Z=svd(U)
println("\nSVD decomposition=\r")
display(Z)
println("\nUapprox based on the first and most energetic mode=\r")
Ũ=(Z.U[:,1]*Z.S[1])*Z.Vt[1,:]'
println("\nŨ=\r")
display(Ũ)
