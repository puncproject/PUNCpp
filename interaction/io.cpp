// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

/**
 * @file		io.c
 * @brief		Functions for parsing options from ini file
 */

#include "io.h"

namespace punc {

bool split_suffix(const string &in, string &body, string &suffix, const vector<string> &suffixes){
    for(auto &s : suffixes){
        size_t pos = in.rfind(s);
        if(pos != string::npos){
            suffix = s;
            body = in.substr(0,pos);
            return true;
        }
    }
    return false;
}

vector<Species> read_species(const Options &opt, const Mesh &mesh){

    PhysicalConstants constants;

    vector<double> charge, mass, density, thermal;
    vector<string> charge_suffix, mass_suffix, thermal_suffix;
    opt.get_repeated("species.charge"     , charge , 0, {"C", "e", ""}, charge_suffix);
    opt.get_repeated("species.mass"       , mass   , 0, {"kg", "me", "amu", ""}, mass_suffix);
    opt.get_repeated("species.density"    , density, 0);
    opt.get_repeated("species.temperature", thermal, 0, {"K", "m/s", "eV", ""}, thermal_suffix);

    size_t nSpecies = charge.size();
    if(mass.size()    != nSpecies
    || density.size() != nSpecies
    || thermal.size() != nSpecies){
        
        cerr << "Unequal number of species.charge, species.mass, "
             << "species.density and species.temperature specified" << endl;
        exit(1);
    }

    for(size_t s=0; s<nSpecies; s++){
        if(charge_suffix[s]=="e") charge[s] *= constants.e;
        if(mass_suffix[s]=="me")  mass[s]   *= constants.m_e;
        if(mass_suffix[s]=="amu") mass[s]   *= constants.amu;
        if(thermal_suffix[s]=="eV"){
            thermal[s] *= constants.e/constants.k_B;
            thermal[s] = sqrt(constants.k_B*thermal[s]/mass[s]);
        }
        if(thermal_suffix[s]=="K" || thermal_suffix[s]==""){
            thermal[s] = sqrt(constants.k_B*thermal[s]/mass[s]);
        }
    }

    vector<string> distribution(nSpecies, "maxwellian");
    opt.get_repeated("species.distribution", distribution, nSpecies, true);

    vector<double> amount;
    vector<string> amount_suffix(nSpecies, "in total");
    opt.get_repeated("species.amount", amount, nSpecies,
                     {"in total", "per cell", "per volume", "phys per sim", ""},
                     amount_suffix);

    vector<double> kappa(nSpecies, 4);
    opt.get_repeated("species.kappa", kappa, nSpecies, true);

    vector<double> alpha(nSpecies, 0);
    opt.get_repeated("species.alpha", alpha, nSpecies, true);

    vector<vector<double>> vdrift(nSpecies, vector<double>(mesh.dim));
    opt.get_repeated_vector("species.vdrift", vdrift, mesh.dim, nSpecies, true);

    vector<Species> species;

    for(size_t s=0; s<nSpecies; s++){

        std::shared_ptr<Pdf> pdf = std::make_shared<UniformPosition>(mesh);
        std::shared_ptr<Pdf> vdf;

        if(distribution[s]=="maxwellian"){
            vdf = std::make_shared<Maxwellian>(thermal[s], vdrift[s]);
        } else if (distribution[s]=="kappa") {
            vdf = std::make_shared<Kappa>(thermal[s], vdrift[s], kappa[s]);
        }else if (distribution[s] == "cairns"){
            vdf = std::make_shared<Cairns>(thermal[s], vdrift[s], alpha[s]);
        }else if (distribution[s] == "kappa-cairns"){
            vdf = std::make_shared<KappaCairns>(thermal[s], vdrift[s], kappa[s], alpha[s]);
        } else {
            cerr << "species.distribution must be one of: "
                 << "\"maxwellian\", \"kappa\", \"cairns\", \"kappa-cairns\""
                 << endl;
            exit(1);
        }

        ParticleAmountType type = ParticleAmountType::in_total;
        if(amount_suffix[s]=="per cell"){
            type = ParticleAmountType::per_cell;
        } else if(amount_suffix[s]=="per volume"){
            type = ParticleAmountType::per_volume;
        } else if(amount_suffix[s]=="phys per sim"){
            type = ParticleAmountType::phys_per_sim;
        }

        species.emplace_back(charge[s], mass[s], density[s], amount[s],
                             type, mesh, pdf, vdf);
    }

    return species;
}

} // namespace punc
