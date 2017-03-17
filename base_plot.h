// From Nicolas

#ifndef base_plot_H
#define base_plot_H

#include <iostream>
using namespace std;
#include <string>
#include <sstream>

#include <TH1.h>
#include <TGraph.h>
#include <TTree.h>
#include <TDirectory.h>

TH1D* GetHistoWeight(TTree* t, string variable, int nbins, double xmin, double xmax, string cut, string weight, string name){
	string sxmin, sxmax, snbins;
	stringstream ss[3];

	ss[0] << xmin;
	ss[0] >> sxmin;
	ss[1] << xmax;
	ss[1] >> sxmax;
	ss[2] << nbins;
	ss[2] >> snbins;

	string variablenew = variable + " >> h(" + snbins + "," + sxmin + "," + sxmax + ")";

	string cutnew = weight + " * (" + cut + ")";

	t->Draw(variablenew.c_str(), cutnew.c_str());
	TH1D *histo = (TH1D*)gDirectory->Get("h");

	if (histo->GetEntries()==0) return histo;

	double underflow = histo->GetBinContent(0);
	//cout << "underflow="<<underflow<<endl;
	double val = 0;
	if (underflow>0) {
		val = histo->GetBinContent(1);
		histo->SetBinContent(1, val+underflow);
		histo->SetBinContent(0, 0);
	}
	double overflow = histo->GetBinContent(nbins+1);
	if (overflow>0) {
		val = histo->GetBinContent(nbins);
		histo->SetBinContent(nbins+1, 0);
		histo->SetBinContent(nbins, val+overflow);
	}

	//cout << "Area="<<histo->Integral()<<endl;
	//cout << "Nevents="<<histo->GetEntries()<<endl;
	histo->SetName(name.c_str());
	histo->SetTitle(name.c_str());

	return histo;
}

TGraph * GetEffSvsEffB(TTree* Signal, TTree* Background, string presel_sig, string presel_bkg, string var, double valmin, double valmax, string weight, int npoints, string TitleGraph){
	int nbins = npoints;

	double* Eff_sg = new double[nbins];
	double* Eff_err_sg = new double[nbins];

	double* Eff_bg = new double[nbins];
	double* Eff_err_bg = new double[nbins];

	//stringstream ss[nbins+1];
	string svalcut;
	string scut;

	string presel_sg = presel_sig;
	string presel_bg = presel_bkg;

	TH1D* Histo_sg = GetHistoWeight(Signal, var, npoints, valmin, valmax, presel_sg, weight, "sg");
	Histo_sg->SetName("sg");
	double denom_sg = Histo_sg->Integral();
	cout << "denom_sg=" << denom_sg<< endl;

	TH1D* Histo_bg = GetHistoWeight(Background, var, npoints, valmin, valmax, presel_bg, weight, "bg");
	Histo_bg->SetName("bg");
	double denom_bg = Histo_bg->Integral();
	cout << "denom_bg=" << denom_bg<< endl;

	Eff_sg[0]=1.;
	Eff_bg[0]=1.;

	for (int i=1; i<=nbins; i++){
		double valcut = valmax*((double)i)/((double)nbins) + valmin*(1-((double)i)/((double)nbins));

		//ss[i] << valcut;
		//ss[i] >> svalcut;

		double num_sg = Histo_sg->Integral(i, nbins);
		double num_bg = Histo_bg->Integral(i, nbins);

		Eff_sg[i] = num_sg/denom_sg;
		Eff_err_sg[i] = Eff_sg[i] * (sqrt(num_sg)/num_sg + sqrt(denom_sg)/denom_sg);

		Eff_bg[i] = num_bg/denom_bg;
		Eff_err_bg[i] = Eff_bg[i] * (sqrt(num_bg)/num_bg + sqrt(denom_bg)/denom_bg);

		//cout << "i="<<i << " "<<var<< "<" <<svalcut<<"   effS="<<Eff_sg[i]<<" +/- "<< Eff_err_sg[i]<<"   effB="<< Eff_bg[i]<<" +/- "<< Eff_err_bg[i]<<endl;

	}

	TGraph* GraphEff = new TGraph(nbins+1, Eff_bg, Eff_sg);
	GraphEff->SetName(TitleGraph.c_str());
	GraphEff->SetTitle(TitleGraph.c_str());

	return GraphEff;

}

#endif
