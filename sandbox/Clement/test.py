import dolfin as df
# from punc import load_mesh


fname = "/home/diako/Documents/cpp/punc/mesh/2D/circle_in_square_res1" 
mesh = df.Mesh(fname + ".xml")

tdim = mesh.topology().dim()
gdim = mesh.geometry().dim()
cv = df.CellVolume(mesh)

V = df.FunctionSpace(mesh, 'CG', 1)
Q = df.FunctionSpace(mesh, 'DG', 0)
p, q = df.TrialFunction(Q), df.TestFunction(Q)
v = df.TestFunction(V)

ones = df.assemble(
    (1. / cv) * df.inner(df.Constant(1), q) * df.dx)

dX = df.dx(metadata={'form_compiler_parameters': {
            'quadrature_degree': 1, 'quadrature_scheme': 'vertex'}})


A = df.assemble(df.Constant(tdim+1)*df.inner(p, v)*dX)

Av = df.Function(V).vector()
A.mult(ones, Av)

print(Av.size())
print(A.size(0))
print(A.size(1))
print(ones.size())

Av = df.as_backend_type(Av).vec()
Av.reciprocal()
mat = df.as_backend_type(A).mat()
mat.diagonalScale(L=Av)

# print(A)

e_exp = df.Expression('((phi_1-phi_2)*r_1*r_2/( (r_2-r_1) * pow(x[0]*x[0]+x[1]*x[1]+x[2]*x[2], 0.5) )) + phi_1-r_2*(phi_1-phi_2)/(r_2-r_1)', phi_1=1.0,phi_2=0.0,r_1=0.02,r_2=0.2,degree=1)

e_dg0 = df.interpolate(e_exp,Q)

e_field = df.Function(V)
A.mult(e_dg0.vector(), e_field.vector())
df.as_backend_type(e_field.vector()).update_ghost_values()

df.File('phi.pvd') << e_field
df.File('phiDG.pvd') << e_dg0

