#include "create.h"

using namespace dolfin;

void Source::eval(Array<double> &values, const Array<double> &x) const
{
    double dx = x[0] - 3.14;
    double dy = x[1] - 3.14;
    values[0] = 10 * exp(-(dx * dx + dy * dy) / 0.02);
}

CreateObject::CreateObject(const double& r, const std::vector<double>&s,
                           double tol):SubDomain(),r(r), s(s), tol(tol)
{
    std::function<bool(const Array<double>&)> func = \
    [r,s,tol](const Array<double>& x)->bool
    {
        double dot = 0.0;
        for(std::size_t i = 0; i<s.size(); ++i)
        {
            dot += (x[i]-s[i])*(x[i]-s[i]);
        }
        return dot <= r*r+tol;
    };
    this->func = func;

}

bool CreateObject::inside(const Array<double>& x, bool on_boundary) const
{
    return on_boundary && func(x);
}
