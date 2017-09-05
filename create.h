#ifndef CREATE_H
#define CREATE_H

#include <iostream>
#include <dolfin.h>
#include <algorithm>

using namespace dolfin;

class Source : public Expression
{
    void eval(Array<double> &values, const Array<double> &x) const;
};

class CreateObject : public SubDomain
{
public:
    const double& r;
    const std::vector<double>& s;
    double tol;
    std::function<bool(const Array<double>&)> func;

    CreateObject(const double& r, const std::vector<double>&s, double tol=1e-4);

    bool inside(const Array<double>& x, bool on_boundary) const;
};

#endif
