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

vector<Species> read_species(po::variables_map options, const Mesh &mesh){

    PhysicalConstants constants;

    vector<double> charge  = options["species.charge"].as<vector<double>>();
    vector<double> mass    = options["species.mass"].as<vector<double>>();
    vector<double> thermal = options["species.thermal"].as<vector<double>>();
    vector<double> density = options["species.density"].as<vector<double>>();

    // vector<double> density = options.get<vector<double>>("species.density");
    // auto density = options.get<vector<double>>("species.density");
    // vector<double> density;
    // options.get("species.density", density);

    size_t nSpecies = charge.size();
    if(mass.size()    != nSpecies
    || density.size() != nSpecies
    || thermal.size() != nSpecies){
        
        cerr << "Inconsistent number of species specified. "
             << "Check species.charge, species.mass, species.density and species.thermal" << endl;
        exit(1);
    }

    for(size_t s=0; s<nSpecies; s++){
        charge[s] *= constants.e;
        mass[s] *= constants.m_e;
    }

    vector<string> distribution = get_repeated<string>(options, "species.distribution", nSpecies, "maxwellian");
    vector<int> npc             = get_repeated<int>(options, "species.npc", nSpecies, 0);
    vector<int> num             = get_repeated<int>(options, "species.num", nSpecies, 0);
    vector<double> kappa        = get_repeated<double>(options, "species.kappa", nSpecies, 0);
    vector<double> alpha        = get_repeated<double>(options, "species.alpha", nSpecies, 0);
    vector<vector<double>> vd   = get_repeated_vector<double>(options, "species.vdrift", nSpecies, mesh.dim, vector<double>(mesh.dim, 0));

    vector<Species> species;

    for(size_t s=0; s<charge.size(); s++){

        std::shared_ptr<Pdf> pdf = std::make_shared<UniformPosition>(mesh);
        std::shared_ptr<Pdf> vdf;

        if(distribution[s]=="maxwellian"){
            vdf = std::make_shared<Maxwellian>(thermal[s], vd[s]);
        } else if (distribution[s]=="kappa") {
            vdf = std::make_shared<Kappa>(thermal[s], vd[s], kappa[s]);
        }else if (distribution[s] == "cairns"){
            vdf = std::make_shared<Cairns>(thermal[s], vd[s], alpha[s]);
        }else if (distribution[s] == "kappa-cairns"){
            vdf = std::make_shared<KappaCairns>(thermal[s], vd[s], kappa[s], alpha[s]);
        } else {
            cerr << "Unsupported velocity distribution: ";
            cerr << distribution[s] << endl;
            exit(1);
        }

        species.emplace_back(charge[s], mass[s], density[s], num[s],
                ParticleAmountType::in_total, mesh, pdf, vdf);
    }

    return species;
}

} // namespace punc
