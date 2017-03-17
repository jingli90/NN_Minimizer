//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Mar  2 14:54:55 2017 by ROOT version 6.02/13
// from TTree Tree/Tree
// found on file: /afs/cern.ch/work/j/jing/public/IPHCNtuple/fromXavier/trees_from_Xavier_20170302/ttHToNonbb.root
//////////////////////////////////////////////////////////

#ifndef base_h
#define base_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"
#include "TLorentzVector.h"

class base {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
//   Int_t           mc_event;
   Float_t         weight;
//   Float_t         weightfake;
//   Float_t         weightflip;
//   Float_t         mc_weight;
//   Float_t         weight_scale_muF0p5;
//   Float_t         weight_scale_muF2;
//   Float_t         weight_scale_muR0p5;
//   Float_t         weight_scale_muR2;
//   Float_t         weight_csv_down;
//   Float_t         weight_csv_up;
//   vector<float>   *weights_pdf;
//   vector<string>  *ids_pdf;
//   Float_t         PV_weight;
//   Int_t           mc_3l_category;
//   Int_t           mc_ttbar_decay;
//   Int_t           mc_boson_decay;
   Int_t           mc_ttZhypAllowed;
//   Int_t           mc_nJets25;
//   Int_t           mc_nBtagJets25;
//   Int_t           mc_nMediumBtagJets25;
//   Int_t           mc_nNonBtagJets25;
//   Int_t           catJets;
//   Char_t          is_2lss_TTH_SR;
   Char_t          is_3l_TTH_SR;
//   Char_t          is_emu_TT_CR;
//   Char_t          is_3l_WZrel_CR;
//   Char_t          is_3l_TTZ_CR;
//   Int_t           is_2bTight;
//   Float_t         is_2bTight_float;
//   Char_t          is_3l_TZQ_SR;
//   Int_t           cat_ee;
//   Int_t           cat_ee_fake;
//   Int_t           cat_ee_flip;
//   Int_t           cat_em;
//   Int_t           cat_em_fake;
//   Int_t           cat_em_flip;
//   Int_t           cat_mm;
//   Int_t           cat_mm_fake;
//   Int_t           cat_2ltau;
//   Int_t           cat_3l;
//   Int_t           cat_3l_fake;
//   Char_t          cat_HtoWW;
//   Char_t          cat_HtoZZ;
//   Char_t          cat_Htott;
//   Char_t          is_trigger;
   Float_t         max_Lep_eta;
   Float_t         MT_met_lep1;
   Float_t         nJet25_Recl;
   Float_t         mindr_lep1_jet;
   Float_t         mindr_lep2_jet;
   Float_t         LepGood_conePt0;
   Float_t         LepGood_conePt1;
//   Float_t         met;
//   Float_t         mhtJet25_Recl;
//   Float_t         avg_dr_jet;
//   Float_t         signal_2lss_TT_MVA;
//   Float_t         signal_2lss_TTV_MVA;
//   Float_t         signal_3l_TT_MVA;
//   Float_t         signal_3l_TTV_MVA;
//   Int_t           multilepton_Lepton1_Id;
//   TLorentzVector  *multilepton_Lepton1_P4;
//   Float_t         multilepton_Lepton1_DeltaR_Matched;
//   Int_t           multilepton_Lepton1_Label_Matched;
//   Int_t           multilepton_Lepton1_Id_Matched;
//   TLorentzVector  *multilepton_Lepton1_P4_Matched;
//   Int_t           multilepton_Lepton2_Id;
//   TLorentzVector  *multilepton_Lepton2_P4;
//   Float_t         multilepton_Lepton2_DeltaR_Matched;
//   Int_t           multilepton_Lepton2_Label_Matched;
//   Int_t           multilepton_Lepton2_Id_Matched;
//   TLorentzVector  *multilepton_Lepton2_P4_Matched;
//   Int_t           multilepton_Lepton3_Id;
//   TLorentzVector  *multilepton_Lepton3_P4;
//   Float_t         multilepton_Lepton3_DeltaR_Matched;
//   Int_t           multilepton_Lepton3_Label_Matched;
//   Int_t           multilepton_Lepton3_Id_Matched;
//   TLorentzVector  *multilepton_Lepton3_P4_Matched;
//   Int_t           multilepton_Lepton4_Id;
//   TLorentzVector  *multilepton_Lepton4_P4;
//   Float_t         multilepton_Lepton4_DeltaR_Matched;
//   Int_t           multilepton_Lepton4_Label_Matched;
//   Int_t           multilepton_Lepton4_Id_Matched;
//   TLorentzVector  *multilepton_Lepton4_P4_Matched;
//   Int_t           multilepton_Bjet1_Id;
//   TLorentzVector  *multilepton_Bjet1_P4;
//   Float_t         multilepton_Bjet1_CSV;
//   Float_t         multilepton_Bjet1_JEC_Up;
//   Float_t         multilepton_Bjet1_JEC_Down;
//   Float_t         multilepton_Bjet1_JER_Up;
//   Float_t         multilepton_Bjet1_JER_Down;
//   Float_t         multilepton_Bjet1_DeltaR_Matched;
//   Int_t           multilepton_Bjet1_Label_Matched;
//   Int_t           multilepton_Bjet1_Id_Matched;
//   TLorentzVector  *multilepton_Bjet1_P4_Matched;
//   Int_t           multilepton_Bjet2_Id;
//   TLorentzVector  *multilepton_Bjet2_P4;
//   Float_t         multilepton_Bjet2_CSV;
//   Float_t         multilepton_Bjet2_JEC_Up;
//   Float_t         multilepton_Bjet2_JEC_Down;
//   Float_t         multilepton_Bjet2_JER_Up;
//   Float_t         multilepton_Bjet2_JER_Down;
//   Float_t         multilepton_Bjet2_DeltaR_Matched;
//   Int_t           multilepton_Bjet2_Label_Matched;
//   Int_t           multilepton_Bjet2_Id_Matched;
//   TLorentzVector  *multilepton_Bjet2_P4_Matched;
//   Int_t           multilepton_JetHighestPt1_Id;
//   TLorentzVector  *multilepton_JetHighestPt1_P4;
//   Float_t         multilepton_JetHighestPt1_CSV;
//   Float_t         multilepton_JetHighestPt1_JEC_Up;
//   Float_t         multilepton_JetHighestPt1_JEC_Down;
//   Float_t         multilepton_JetHighestPt1_JER_Up;
//   Float_t         multilepton_JetHighestPt1_JER_Down;
//   Int_t           multilepton_JetHighestPt2_Id;
//   TLorentzVector  *multilepton_JetHighestPt2_P4;
//   Float_t         multilepton_JetHighestPt2_CSV;
//   Float_t         multilepton_JetHighestPt2_JEC_Up;
//   Float_t         multilepton_JetHighestPt2_JEC_Down;
//   Float_t         multilepton_JetHighestPt2_JER_Up;
//   Float_t         multilepton_JetHighestPt2_JER_Down;
//   Int_t           multilepton_JetClosestMw1_Id;
//   TLorentzVector  *multilepton_JetClosestMw1_P4;
//   Float_t         multilepton_JetClosestMw1_CSV;
//   Float_t         multilepton_JetClosestMw1_JEC_Up;
//   Float_t         multilepton_JetClosestMw1_JEC_Down;
//   Float_t         multilepton_JetClosestMw1_JER_Up;
//   Float_t         multilepton_JetClosestMw1_JER_Down;
//   Int_t           multilepton_JetClosestMw2_Id;
//   TLorentzVector  *multilepton_JetClosestMw2_P4;
//   Float_t         multilepton_JetClosestMw2_CSV;
//   Float_t         multilepton_JetClosestMw2_JEC_Up;
//   Float_t         multilepton_JetClosestMw2_JEC_Down;
//   Float_t         multilepton_JetClosestMw2_JER_Up;
//   Float_t         multilepton_JetClosestMw2_JER_Down;
//   Int_t           multilepton_JetLowestMjj1_Id;
//   TLorentzVector  *multilepton_JetLowestMjj1_P4;
//   Float_t         multilepton_JetLowestMjj1_CSV;
//   Float_t         multilepton_JetLowestMjj1_JEC_Up;
//   Float_t         multilepton_JetLowestMjj1_JEC_Down;
//   Float_t         multilepton_JetLowestMjj1_JER_Up;
//   Float_t         multilepton_JetLowestMjj1_JER_Down;
//   Int_t           multilepton_JetLowestMjj2_Id;
//   TLorentzVector  *multilepton_JetLowestMjj2_P4;
//   Float_t         multilepton_JetLowestMjj2_CSV;
//   Float_t         multilepton_JetLowestMjj2_JEC_Up;
//   Float_t         multilepton_JetLowestMjj2_JEC_Down;
//   Float_t         multilepton_JetLowestMjj2_JER_Up;
//   Float_t         multilepton_JetLowestMjj2_JER_Down;
//   Int_t           multilepton_JetHighestPt1_2ndPair_Id;
//   TLorentzVector  *multilepton_JetHighestPt1_2ndPair_P4;
//   Float_t         multilepton_JetHighestPt1_2ndPair_CSV;
//   Float_t         multilepton_JetHighestPt1_2ndPair_JEC_Up;
//   Float_t         multilepton_JetHighestPt1_2ndPair_JEC_Down;
//   Float_t         multilepton_JetHighestPt1_2ndPair_JER_Up;
//   Float_t         multilepton_JetHighestPt1_2ndPair_JER_Down;
//   Int_t           multilepton_JetHighestPt2_2ndPair_Id;
//   TLorentzVector  *multilepton_JetHighestPt2_2ndPair_P4;
//   Float_t         multilepton_JetHighestPt2_2ndPair_CSV;
//   Float_t         multilepton_JetHighestPt2_2ndPair_JEC_Up;
//   Float_t         multilepton_JetHighestPt2_2ndPair_JEC_Down;
//   Float_t         multilepton_JetHighestPt2_2ndPair_JER_Up;
//   Float_t         multilepton_JetHighestPt2_2ndPair_JER_Down;
//   Int_t           multilepton_JetClosestMw1_2ndPair_Id;
//   TLorentzVector  *multilepton_JetClosestMw1_2ndPair_P4;
//   Float_t         multilepton_JetClosestMw1_2ndPair_CSV;
//   Float_t         multilepton_JetClosestMw1_2ndPair_JEC_Up;
//   Float_t         multilepton_JetClosestMw1_2ndPair_JEC_Down;
//   Float_t         multilepton_JetClosestMw1_2ndPair_JER_Up;
//   Float_t         multilepton_JetClosestMw1_2ndPair_JER_Down;
//   Int_t           multilepton_JetClosestMw2_2ndPair_Id;
//   TLorentzVector  *multilepton_JetClosestMw2_2ndPair_P4;
//   Float_t         multilepton_JetClosestMw2_2ndPair_CSV;
//   Float_t         multilepton_JetClosestMw2_2ndPair_JEC_Up;
//   Float_t         multilepton_JetClosestMw2_2ndPair_JEC_Down;
//   Float_t         multilepton_JetClosestMw2_2ndPair_JER_Up;
//   Float_t         multilepton_JetClosestMw2_2ndPair_JER_Down;
//   Int_t           multilepton_JetLowestMjj1_2ndPair_Id;
//   TLorentzVector  *multilepton_JetLowestMjj1_2ndPair_P4;
//   Float_t         multilepton_JetLowestMjj1_2ndPair_CSV;
//   Float_t         multilepton_JetLowestMjj1_2ndPair_JEC_Up;
//   Float_t         multilepton_JetLowestMjj1_2ndPair_JEC_Down;
//   Float_t         multilepton_JetLowestMjj1_2ndPair_JER_Up;
//   Float_t         multilepton_JetLowestMjj1_2ndPair_JER_Down;
//   Int_t           multilepton_JetLowestMjj2_2ndPair_Id;
//   TLorentzVector  *multilepton_JetLowestMjj2_2ndPair_P4;
//   Float_t         multilepton_JetLowestMjj2_2ndPair_CSV;
//   Float_t         multilepton_JetLowestMjj2_2ndPair_JEC_Up;
//   Float_t         multilepton_JetLowestMjj2_2ndPair_JEC_Down;
//   Float_t         multilepton_JetLowestMjj2_2ndPair_JER_Up;
//   Float_t         multilepton_JetLowestMjj2_2ndPair_JER_Down;
//   Int_t           multilepton_h0_Id;
//   TLorentzVector  *multilepton_h0_P4;
//   Int_t           multilepton_h0_Label;
//   Int_t           multilepton_t1_Id;
//   TLorentzVector  *multilepton_t1_P4;
//   Int_t           multilepton_t1_Label;
//   Int_t           multilepton_t2_Id;
//   TLorentzVector  *multilepton_t2_P4;
//   Int_t           multilepton_t2_Label;
//   TLorentzVector  *multilepton_mET;
//   Double_t        multilepton_mETcov00;
//   Double_t        multilepton_mETcov01;
//   Double_t        multilepton_mETcov10;
//   Double_t        multilepton_mETcov11;
//   Float_t         multilepton_mHT;
//   TLorentzVector  *multilepton_Ptot;

   // List of branches
//   TBranch        *b_mc_event;   //!
   TBranch        *b_weight;   //!
//   TBranch        *b_weightfake;   //!
//   TBranch        *b_weightflip;   //!
//   TBranch        *b_mc_weight;   //!
//   TBranch        *b_weight_scale_muF0p5;   //!
//   TBranch        *b_weight_scale_muF2;   //!
//   TBranch        *b_weight_scale_muR0p5;   //!
//   TBranch        *b_weight_scale_muR2;   //!
//   TBranch        *b_weight_csv_down;   //!
//   TBranch        *b_weight_csv_up;   //!
//   TBranch        *b_weights_pdf;   //!
//   TBranch        *b_ids_pdf;   //!
//   TBranch        *b_PV_weight;   //!
//   TBranch        *b_mc_3l_category;   //!
//   TBranch        *b_mc_ttbar_decay;   //!
//   TBranch        *b_mc_boson_decay;   //!
   TBranch        *b_mc_ttZhypAllowed;   //!
//   TBranch        *b_mc_nJets25;   //!
//   TBranch        *b_mc_nBtagJets25;   //!
//   TBranch        *b_mc_nMediumBtagJets25;   //!
//   TBranch        *b_mc_nNonBtagJets25;   //!
//   TBranch        *b_catJets;   //!
//   TBranch        *b_is_2lss_TTH_SR;   //!
   TBranch        *b_is_3l_TTH_SR;   //!
//   TBranch        *b_is_emu_TT_CR;   //!
//   TBranch        *b_is_3l_WZrel_CR;   //!
//   TBranch        *b_is_3l_TTZ_CR;   //!
//   TBranch        *b_is_2bTight;   //!
//   TBranch        *b_is_2bTight_float;   //!
//   TBranch        *b_is_3l_TZQ_SR;   //!
//   TBranch        *b_cat_ee;   //!
//   TBranch        *b_cat_ee_fake;   //!
//   TBranch        *b_cat_ee_flip;   //!
//   TBranch        *b_cat_em;   //!
//   TBranch        *b_cat_em_fake;   //!
//   TBranch        *b_cat_em_flip;   //!
//   TBranch        *b_cat_mm;   //!
//   TBranch        *b_cat_mm_fake;   //!
//   TBranch        *b_cat_2ltau;   //!
//   TBranch        *b_cat_3l;   //!
//   TBranch        *b_cat_3l_fake;   //!
//   TBranch        *b_cat_HtoWW;   //!
//   TBranch        *b_cat_HtoZZ;   //!
//   TBranch        *b_cat_Htott;   //!
//   TBranch        *b_is_trigger;   //!
   TBranch        *b_max_Lep_eta;   //!
   TBranch        *b_MT_met_lep1;   //!
   TBranch        *b_nJet25_Recl;   //!
   TBranch        *b_mindr_lep1_jet;   //!
   TBranch        *b_mindr_lep2_jet;   //!
   TBranch        *b_LepGood_conePt0;   //!
   TBranch        *b_LepGood_conePt1;   //!
//   TBranch        *b_met;   //!
//   TBranch        *b_mhtJet25_Recl;   //!
//   TBranch        *b_avg_dr_jet;   //!
//   TBranch        *b_signal_2lss_TT_MVA;   //!
//   TBranch        *b_signal_2lss_TTV_MVA;   //!
//   TBranch        *b_signal_3l_TT_MVA;   //!
//   TBranch        *b_signal_3l_TTV_MVA;   //!
//   TBranch        *b_multilepton_Lepton1_Id;   //!
//   TBranch        *b_multilepton_Lepton1_P4;   //!
//   TBranch        *b_multilepton_Lepton1_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Lepton1_Label_Matched;   //!
//   TBranch        *b_multilepton_Lepton1_Id_Matched;   //!
//   TBranch        *b_multilepton_Lepton1_P4_Matched;   //!
//   TBranch        *b_multilepton_Lepton2_Id;   //!
//   TBranch        *b_multilepton_Lepton2_P4;   //!
//   TBranch        *b_multilepton_Lepton2_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Lepton2_Label_Matched;   //!
//   TBranch        *b_multilepton_Lepton2_Id_Matched;   //!
//   TBranch        *b_multilepton_Lepton2_P4_Matched;   //!
//   TBranch        *b_multilepton_Lepton3_Id;   //!
//   TBranch        *b_multilepton_Lepton3_P4;   //!
//   TBranch        *b_multilepton_Lepton3_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Lepton3_Label_Matched;   //!
//   TBranch        *b_multilepton_Lepton3_Id_Matched;   //!
//   TBranch        *b_multilepton_Lepton3_P4_Matched;   //!
//   TBranch        *b_multilepton_Lepton4_Id;   //!
//   TBranch        *b_multilepton_Lepton4_P4;   //!
//   TBranch        *b_multilepton_Lepton4_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Lepton4_Label_Matched;   //!
//   TBranch        *b_multilepton_Lepton4_Id_Matched;   //!
//   TBranch        *b_multilepton_Lepton4_P4_Matched;   //!
//   TBranch        *b_multilepton_Bjet1_Id;   //!
//   TBranch        *b_multilepton_Bjet1_P4;   //!
//   TBranch        *b_multilepton_Bjet1_CSV;   //!
//   TBranch        *b_multilepton_Bjet1_JEC_Up;   //!
//   TBranch        *b_multilepton_Bjet1_JEC_Down;   //!
//   TBranch        *b_multilepton_Bjet1_JER_Up;   //!
//   TBranch        *b_multilepton_Bjet1_JER_Down;   //!
//   TBranch        *b_multilepton_Bjet1_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Bjet1_Label_Matched;   //!
//   TBranch        *b_multilepton_Bjet1_Id_Matched;   //!
//   TBranch        *b_multilepton_Bjet1_P4_Matched;   //!
//   TBranch        *b_multilepton_Bjet2_Id;   //!
//   TBranch        *b_multilepton_Bjet2_P4;   //!
//   TBranch        *b_multilepton_Bjet2_CSV;   //!
//   TBranch        *b_multilepton_Bjet2_JEC_Up;   //!
//   TBranch        *b_multilepton_Bjet2_JEC_Down;   //!
//   TBranch        *b_multilepton_Bjet2_JER_Up;   //!
//   TBranch        *b_multilepton_Bjet2_JER_Down;   //!
//   TBranch        *b_multilepton_Bjet2_DeltaR_Matched;   //!
//   TBranch        *b_multilepton_Bjet2_Label_Matched;   //!
//   TBranch        *b_multilepton_Bjet2_Id_Matched;   //!
//   TBranch        *b_multilepton_Bjet2_P4_Matched;   //!
//   TBranch        *b_multilepton_JetHighestPt1_Id;   //!
//   TBranch        *b_multilepton_JetHighestPt1_P4;   //!
//   TBranch        *b_multilepton_JetHighestPt1_CSV;   //!
//   TBranch        *b_multilepton_JetHighestPt1_JEC_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt1_JEC_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt1_JER_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt1_JER_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt2_Id;   //!
//   TBranch        *b_multilepton_JetHighestPt2_P4;   //!
//   TBranch        *b_multilepton_JetHighestPt2_CSV;   //!
//   TBranch        *b_multilepton_JetHighestPt2_JEC_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt2_JEC_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt2_JER_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt2_JER_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw1_Id;   //!
//   TBranch        *b_multilepton_JetClosestMw1_P4;   //!
//   TBranch        *b_multilepton_JetClosestMw1_CSV;   //!
//   TBranch        *b_multilepton_JetClosestMw1_JEC_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw1_JEC_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw1_JER_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw1_JER_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw2_Id;   //!
//   TBranch        *b_multilepton_JetClosestMw2_P4;   //!
//   TBranch        *b_multilepton_JetClosestMw2_CSV;   //!
//   TBranch        *b_multilepton_JetClosestMw2_JEC_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw2_JEC_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw2_JER_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw2_JER_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_Id;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_P4;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_CSV;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_JEC_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_JEC_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_JER_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_JER_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_Id;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_P4;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_CSV;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_JEC_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_JEC_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_JER_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_JER_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt1_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetHighestPt2_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw1_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetClosestMw2_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj1_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_Id;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_P4;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_CSV;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_JEC_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_JEC_Down;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_JER_Up;   //!
//   TBranch        *b_multilepton_JetLowestMjj2_2ndPair_JER_Down;   //!
//   TBranch        *b_multilepton_h0_Id;   //!
//   TBranch        *b_multilepton_h0_P4;   //!
//   TBranch        *b_multilepton_h0_Label;   //!
//   TBranch        *b_multilepton_t1_Id;   //!
//   TBranch        *b_multilepton_t1_P4;   //!
//   TBranch        *b_multilepton_h0_Label;   //!
//   TBranch        *b_multilepton_t2_Id;   //!
//   TBranch        *b_multilepton_t2_P4;   //!
//   TBranch        *b_multilepton_h0_Label;   //!
//   TBranch        *b_multilepton_mET;   //!
//   TBranch        *b_multilepton_mETcov00;   //!
//   TBranch        *b_multilepton_mETcov01;   //!
//   TBranch        *b_multilepton_mETcov10;   //!
//   TBranch        *b_multilepton_mETcov11;   //!
//   TBranch        *b_multilepton_mHT;   //!
//   TBranch        *b_multilepton_Ptot;   //!

   base(TTree *tree=0);
   virtual ~base();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   //virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
	virtual Double_t GetVal(TString varName);
};

base::base(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/afs/cern.ch/work/j/jing/public/IPHCNtuple/fromXavier/trees_from_Xavier_20170302/ttHToNonbb.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("/afs/cern.ch/work/j/jing/public/IPHCNtuple/fromXavier/trees_from_Xavier_20170302/ttHToNonbb.root");
      }
      f->GetObject("Tree",tree);

   }
   Init(tree);
}

base::~base()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t base::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t base::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void base::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
//   weights_pdf = 0;
//   ids_pdf = 0;
//   multilepton_Lepton1_P4 = 0;
//   multilepton_Lepton1_P4_Matched = 0;
//   multilepton_Lepton2_P4 = 0;
//   multilepton_Lepton2_P4_Matched = 0;
//   multilepton_Lepton3_P4 = 0;
//   multilepton_Lepton3_P4_Matched = 0;
//   multilepton_Lepton4_P4 = 0;
//   multilepton_Lepton4_P4_Matched = 0;
//   multilepton_Bjet1_P4 = 0;
//   multilepton_Bjet1_P4_Matched = 0;
//   multilepton_Bjet2_P4 = 0;
//   multilepton_Bjet2_P4_Matched = 0;
//   multilepton_JetHighestPt1_P4 = 0;
//   multilepton_JetHighestPt2_P4 = 0;
//   multilepton_JetClosestMw1_P4 = 0;
//   multilepton_JetClosestMw2_P4 = 0;
//   multilepton_JetLowestMjj1_P4 = 0;
//   multilepton_JetLowestMjj2_P4 = 0;
//   multilepton_JetHighestPt1_2ndPair_P4 = 0;
//   multilepton_JetHighestPt2_2ndPair_P4 = 0;
//   multilepton_JetClosestMw1_2ndPair_P4 = 0;
//   multilepton_JetClosestMw2_2ndPair_P4 = 0;
//   multilepton_JetLowestMjj1_2ndPair_P4 = 0;
//   multilepton_JetLowestMjj2_2ndPair_P4 = 0;
//   multilepton_h0_P4 = 0;
//   multilepton_t1_P4 = 0;
//   multilepton_t2_P4 = 0;
//   multilepton_mET = 0;
//   multilepton_Ptot = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

//   fChain->SetBranchAddress("mc_event", &mc_event, &b_mc_event);
   fChain->SetBranchAddress("weight", &weight, &b_weight);
//   fChain->SetBranchAddress("weightfake", &weightfake, &b_weightfake);
//   fChain->SetBranchAddress("weightflip", &weightflip, &b_weightflip);
//   fChain->SetBranchAddress("mc_weight", &mc_weight, &b_mc_weight);
//   fChain->SetBranchAddress("weight_scale_muF0p5", &weight_scale_muF0p5, &b_weight_scale_muF0p5);
//   fChain->SetBranchAddress("weight_scale_muF2", &weight_scale_muF2, &b_weight_scale_muF2);
//   fChain->SetBranchAddress("weight_scale_muR0p5", &weight_scale_muR0p5, &b_weight_scale_muR0p5);
//   fChain->SetBranchAddress("weight_scale_muR2", &weight_scale_muR2, &b_weight_scale_muR2);
//   fChain->SetBranchAddress("weight_csv_down", &weight_csv_down, &b_weight_csv_down);
//   fChain->SetBranchAddress("weight_csv_up", &weight_csv_up, &b_weight_csv_up);
//   fChain->SetBranchAddress("weights_pdf", &weights_pdf, &b_weights_pdf);
//   fChain->SetBranchAddress("ids_pdf", &ids_pdf, &b_ids_pdf);
//   fChain->SetBranchAddress("PV_weight", &PV_weight, &b_PV_weight);
//   fChain->SetBranchAddress("mc_3l_category", &mc_3l_category, &b_mc_3l_category);
//   fChain->SetBranchAddress("mc_ttbar_decay", &mc_ttbar_decay, &b_mc_ttbar_decay);
//   fChain->SetBranchAddress("mc_boson_decay", &mc_boson_decay, &b_mc_boson_decay);
   fChain->SetBranchAddress("mc_ttZhypAllowed", &mc_ttZhypAllowed, &b_mc_ttZhypAllowed);
//   fChain->SetBranchAddress("mc_nJets25", &mc_nJets25, &b_mc_nJets25);
//   fChain->SetBranchAddress("mc_nBtagJets25", &mc_nBtagJets25, &b_mc_nBtagJets25);
//   fChain->SetBranchAddress("mc_nMediumBtagJets25", &mc_nMediumBtagJets25, &b_mc_nMediumBtagJets25);
//   fChain->SetBranchAddress("mc_nNonBtagJets25", &mc_nNonBtagJets25, &b_mc_nNonBtagJets25);
//   fChain->SetBranchAddress("catJets", &catJets, &b_catJets);
//   fChain->SetBranchAddress("is_2lss_TTH_SR", &is_2lss_TTH_SR, &b_is_2lss_TTH_SR);
   fChain->SetBranchAddress("is_3l_TTH_SR", &is_3l_TTH_SR, &b_is_3l_TTH_SR);
//   fChain->SetBranchAddress("is_emu_TT_CR", &is_emu_TT_CR, &b_is_emu_TT_CR);
//   fChain->SetBranchAddress("is_3l_WZrel_CR", &is_3l_WZrel_CR, &b_is_3l_WZrel_CR);
//   fChain->SetBranchAddress("is_3l_TTZ_CR", &is_3l_TTZ_CR, &b_is_3l_TTZ_CR);
//   fChain->SetBranchAddress("is_2bTight", &is_2bTight, &b_is_2bTight);
//   fChain->SetBranchAddress("is_2bTight_float", &is_2bTight_float, &b_is_2bTight_float);
//   fChain->SetBranchAddress("is_3l_TZQ_SR", &is_3l_TZQ_SR, &b_is_3l_TZQ_SR);
//   fChain->SetBranchAddress("cat_ee", &cat_ee, &b_cat_ee);
//   fChain->SetBranchAddress("cat_ee_fake", &cat_ee_fake, &b_cat_ee_fake);
//   fChain->SetBranchAddress("cat_ee_flip", &cat_ee_flip, &b_cat_ee_flip);
//   fChain->SetBranchAddress("cat_em", &cat_em, &b_cat_em);
//   fChain->SetBranchAddress("cat_em_fake", &cat_em_fake, &b_cat_em_fake);
//   fChain->SetBranchAddress("cat_em_flip", &cat_em_flip, &b_cat_em_flip);
//   fChain->SetBranchAddress("cat_mm", &cat_mm, &b_cat_mm);
//   fChain->SetBranchAddress("cat_mm_fake", &cat_mm_fake, &b_cat_mm_fake);
//   fChain->SetBranchAddress("cat_2ltau", &cat_2ltau, &b_cat_2ltau);
//   fChain->SetBranchAddress("cat_3l", &cat_3l, &b_cat_3l);
//   fChain->SetBranchAddress("cat_3l_fake", &cat_3l_fake, &b_cat_3l_fake);
//   fChain->SetBranchAddress("cat_HtoWW", &cat_HtoWW, &b_cat_HtoWW);
//   fChain->SetBranchAddress("cat_HtoZZ", &cat_HtoZZ, &b_cat_HtoZZ);
//   fChain->SetBranchAddress("cat_Htott", &cat_Htott, &b_cat_Htott);
//   fChain->SetBranchAddress("is_trigger", &is_trigger, &b_is_trigger);
   fChain->SetBranchAddress("max_Lep_eta", &max_Lep_eta, &b_max_Lep_eta);
   fChain->SetBranchAddress("MT_met_lep1", &MT_met_lep1, &b_MT_met_lep1);
   fChain->SetBranchAddress("nJet25_Recl", &nJet25_Recl, &b_nJet25_Recl);
   fChain->SetBranchAddress("mindr_lep1_jet", &mindr_lep1_jet, &b_mindr_lep1_jet);
   fChain->SetBranchAddress("mindr_lep2_jet", &mindr_lep2_jet, &b_mindr_lep2_jet);
   fChain->SetBranchAddress("LepGood_conePt0", &LepGood_conePt0, &b_LepGood_conePt0);
   fChain->SetBranchAddress("LepGood_conePt1", &LepGood_conePt1, &b_LepGood_conePt1);
//   fChain->SetBranchAddress("met", &met, &b_met);
//   fChain->SetBranchAddress("mhtJet25_Recl", &mhtJet25_Recl, &b_mhtJet25_Recl);
//   fChain->SetBranchAddress("avg_dr_jet", &avg_dr_jet, &b_avg_dr_jet);
//   fChain->SetBranchAddress("signal_2lss_TT_MVA", &signal_2lss_TT_MVA, &b_signal_2lss_TT_MVA);
//   fChain->SetBranchAddress("signal_2lss_TTV_MVA", &signal_2lss_TTV_MVA, &b_signal_2lss_TTV_MVA);
//   fChain->SetBranchAddress("signal_3l_TT_MVA", &signal_3l_TT_MVA, &b_signal_3l_TT_MVA);
//   fChain->SetBranchAddress("signal_3l_TTV_MVA", &signal_3l_TTV_MVA, &b_signal_3l_TTV_MVA);
//   fChain->SetBranchAddress("multilepton_Lepton1_Id", &multilepton_Lepton1_Id, &b_multilepton_Lepton1_Id);
//   fChain->SetBranchAddress("multilepton_Lepton1_P4", &multilepton_Lepton1_P4, &b_multilepton_Lepton1_P4);
//   fChain->SetBranchAddress("multilepton_Lepton1_DeltaR_Matched", &multilepton_Lepton1_DeltaR_Matched, &b_multilepton_Lepton1_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton1_Label_Matched", &multilepton_Lepton1_Label_Matched, &b_multilepton_Lepton1_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton1_Id_Matched", &multilepton_Lepton1_Id_Matched, &b_multilepton_Lepton1_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton1_P4_Matched", &multilepton_Lepton1_P4_Matched, &b_multilepton_Lepton1_P4_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton2_Id", &multilepton_Lepton2_Id, &b_multilepton_Lepton2_Id);
//   fChain->SetBranchAddress("multilepton_Lepton2_P4", &multilepton_Lepton2_P4, &b_multilepton_Lepton2_P4);
//   fChain->SetBranchAddress("multilepton_Lepton2_DeltaR_Matched", &multilepton_Lepton2_DeltaR_Matched, &b_multilepton_Lepton2_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton2_Label_Matched", &multilepton_Lepton2_Label_Matched, &b_multilepton_Lepton2_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton2_Id_Matched", &multilepton_Lepton2_Id_Matched, &b_multilepton_Lepton2_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton2_P4_Matched", &multilepton_Lepton2_P4_Matched, &b_multilepton_Lepton2_P4_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton3_Id", &multilepton_Lepton3_Id, &b_multilepton_Lepton3_Id);
//   fChain->SetBranchAddress("multilepton_Lepton3_P4", &multilepton_Lepton3_P4, &b_multilepton_Lepton3_P4);
//   fChain->SetBranchAddress("multilepton_Lepton3_DeltaR_Matched", &multilepton_Lepton3_DeltaR_Matched, &b_multilepton_Lepton3_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton3_Label_Matched", &multilepton_Lepton3_Label_Matched, &b_multilepton_Lepton3_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton3_Id_Matched", &multilepton_Lepton3_Id_Matched, &b_multilepton_Lepton3_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton3_P4_Matched", &multilepton_Lepton3_P4_Matched, &b_multilepton_Lepton3_P4_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton4_Id", &multilepton_Lepton4_Id, &b_multilepton_Lepton4_Id);
//   fChain->SetBranchAddress("multilepton_Lepton4_P4", &multilepton_Lepton4_P4, &b_multilepton_Lepton4_P4);
//   fChain->SetBranchAddress("multilepton_Lepton4_DeltaR_Matched", &multilepton_Lepton4_DeltaR_Matched, &b_multilepton_Lepton4_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton4_Label_Matched", &multilepton_Lepton4_Label_Matched, &b_multilepton_Lepton4_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton4_Id_Matched", &multilepton_Lepton4_Id_Matched, &b_multilepton_Lepton4_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Lepton4_P4_Matched", &multilepton_Lepton4_P4_Matched, &b_multilepton_Lepton4_P4_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet1_Id", &multilepton_Bjet1_Id, &b_multilepton_Bjet1_Id);
//   fChain->SetBranchAddress("multilepton_Bjet1_P4", &multilepton_Bjet1_P4, &b_multilepton_Bjet1_P4);
//   fChain->SetBranchAddress("multilepton_Bjet1_CSV", &multilepton_Bjet1_CSV, &b_multilepton_Bjet1_CSV);
//   fChain->SetBranchAddress("multilepton_Bjet1_JEC_Up", &multilepton_Bjet1_JEC_Up, &b_multilepton_Bjet1_JEC_Up);
//   fChain->SetBranchAddress("multilepton_Bjet1_JEC_Down", &multilepton_Bjet1_JEC_Down, &b_multilepton_Bjet1_JEC_Down);
//   fChain->SetBranchAddress("multilepton_Bjet1_JER_Up", &multilepton_Bjet1_JER_Up, &b_multilepton_Bjet1_JER_Up);
//   fChain->SetBranchAddress("multilepton_Bjet1_JER_Down", &multilepton_Bjet1_JER_Down, &b_multilepton_Bjet1_JER_Down);
//   fChain->SetBranchAddress("multilepton_Bjet1_DeltaR_Matched", &multilepton_Bjet1_DeltaR_Matched, &b_multilepton_Bjet1_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet1_Label_Matched", &multilepton_Bjet1_Label_Matched, &b_multilepton_Bjet1_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet1_Id_Matched", &multilepton_Bjet1_Id_Matched, &b_multilepton_Bjet1_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet1_P4_Matched", &multilepton_Bjet1_P4_Matched, &b_multilepton_Bjet1_P4_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet2_Id", &multilepton_Bjet2_Id, &b_multilepton_Bjet2_Id);
//   fChain->SetBranchAddress("multilepton_Bjet2_P4", &multilepton_Bjet2_P4, &b_multilepton_Bjet2_P4);
//   fChain->SetBranchAddress("multilepton_Bjet2_CSV", &multilepton_Bjet2_CSV, &b_multilepton_Bjet2_CSV);
//   fChain->SetBranchAddress("multilepton_Bjet2_JEC_Up", &multilepton_Bjet2_JEC_Up, &b_multilepton_Bjet2_JEC_Up);
//   fChain->SetBranchAddress("multilepton_Bjet2_JEC_Down", &multilepton_Bjet2_JEC_Down, &b_multilepton_Bjet2_JEC_Down);
//   fChain->SetBranchAddress("multilepton_Bjet2_JER_Up", &multilepton_Bjet2_JER_Up, &b_multilepton_Bjet2_JER_Up);
//   fChain->SetBranchAddress("multilepton_Bjet2_JER_Down", &multilepton_Bjet2_JER_Down, &b_multilepton_Bjet2_JER_Down);
//   fChain->SetBranchAddress("multilepton_Bjet2_DeltaR_Matched", &multilepton_Bjet2_DeltaR_Matched, &b_multilepton_Bjet2_DeltaR_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet2_Label_Matched", &multilepton_Bjet2_Label_Matched, &b_multilepton_Bjet2_Label_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet2_Id_Matched", &multilepton_Bjet2_Id_Matched, &b_multilepton_Bjet2_Id_Matched);
//   fChain->SetBranchAddress("multilepton_Bjet2_P4_Matched", &multilepton_Bjet2_P4_Matched, &b_multilepton_Bjet2_P4_Matched);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_Id", &multilepton_JetHighestPt1_Id, &b_multilepton_JetHighestPt1_Id);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_P4", &multilepton_JetHighestPt1_P4, &b_multilepton_JetHighestPt1_P4);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_CSV", &multilepton_JetHighestPt1_CSV, &b_multilepton_JetHighestPt1_CSV);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_JEC_Up", &multilepton_JetHighestPt1_JEC_Up, &b_multilepton_JetHighestPt1_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_JEC_Down", &multilepton_JetHighestPt1_JEC_Down, &b_multilepton_JetHighestPt1_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_JER_Up", &multilepton_JetHighestPt1_JER_Up, &b_multilepton_JetHighestPt1_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_JER_Down", &multilepton_JetHighestPt1_JER_Down, &b_multilepton_JetHighestPt1_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_Id", &multilepton_JetHighestPt2_Id, &b_multilepton_JetHighestPt2_Id);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_P4", &multilepton_JetHighestPt2_P4, &b_multilepton_JetHighestPt2_P4);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_CSV", &multilepton_JetHighestPt2_CSV, &b_multilepton_JetHighestPt2_CSV);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_JEC_Up", &multilepton_JetHighestPt2_JEC_Up, &b_multilepton_JetHighestPt2_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_JEC_Down", &multilepton_JetHighestPt2_JEC_Down, &b_multilepton_JetHighestPt2_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_JER_Up", &multilepton_JetHighestPt2_JER_Up, &b_multilepton_JetHighestPt2_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_JER_Down", &multilepton_JetHighestPt2_JER_Down, &b_multilepton_JetHighestPt2_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_Id", &multilepton_JetClosestMw1_Id, &b_multilepton_JetClosestMw1_Id);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_P4", &multilepton_JetClosestMw1_P4, &b_multilepton_JetClosestMw1_P4);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_CSV", &multilepton_JetClosestMw1_CSV, &b_multilepton_JetClosestMw1_CSV);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_JEC_Up", &multilepton_JetClosestMw1_JEC_Up, &b_multilepton_JetClosestMw1_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_JEC_Down", &multilepton_JetClosestMw1_JEC_Down, &b_multilepton_JetClosestMw1_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_JER_Up", &multilepton_JetClosestMw1_JER_Up, &b_multilepton_JetClosestMw1_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_JER_Down", &multilepton_JetClosestMw1_JER_Down, &b_multilepton_JetClosestMw1_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_Id", &multilepton_JetClosestMw2_Id, &b_multilepton_JetClosestMw2_Id);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_P4", &multilepton_JetClosestMw2_P4, &b_multilepton_JetClosestMw2_P4);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_CSV", &multilepton_JetClosestMw2_CSV, &b_multilepton_JetClosestMw2_CSV);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_JEC_Up", &multilepton_JetClosestMw2_JEC_Up, &b_multilepton_JetClosestMw2_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_JEC_Down", &multilepton_JetClosestMw2_JEC_Down, &b_multilepton_JetClosestMw2_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_JER_Up", &multilepton_JetClosestMw2_JER_Up, &b_multilepton_JetClosestMw2_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_JER_Down", &multilepton_JetClosestMw2_JER_Down, &b_multilepton_JetClosestMw2_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_Id", &multilepton_JetLowestMjj1_Id, &b_multilepton_JetLowestMjj1_Id);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_P4", &multilepton_JetLowestMjj1_P4, &b_multilepton_JetLowestMjj1_P4);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_CSV", &multilepton_JetLowestMjj1_CSV, &b_multilepton_JetLowestMjj1_CSV);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_JEC_Up", &multilepton_JetLowestMjj1_JEC_Up, &b_multilepton_JetLowestMjj1_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_JEC_Down", &multilepton_JetLowestMjj1_JEC_Down, &b_multilepton_JetLowestMjj1_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_JER_Up", &multilepton_JetLowestMjj1_JER_Up, &b_multilepton_JetLowestMjj1_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_JER_Down", &multilepton_JetLowestMjj1_JER_Down, &b_multilepton_JetLowestMjj1_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_Id", &multilepton_JetLowestMjj2_Id, &b_multilepton_JetLowestMjj2_Id);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_P4", &multilepton_JetLowestMjj2_P4, &b_multilepton_JetLowestMjj2_P4);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_CSV", &multilepton_JetLowestMjj2_CSV, &b_multilepton_JetLowestMjj2_CSV);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_JEC_Up", &multilepton_JetLowestMjj2_JEC_Up, &b_multilepton_JetLowestMjj2_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_JEC_Down", &multilepton_JetLowestMjj2_JEC_Down, &b_multilepton_JetLowestMjj2_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_JER_Up", &multilepton_JetLowestMjj2_JER_Up, &b_multilepton_JetLowestMjj2_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_JER_Down", &multilepton_JetLowestMjj2_JER_Down, &b_multilepton_JetLowestMjj2_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_Id", &multilepton_JetHighestPt1_2ndPair_Id, &b_multilepton_JetHighestPt1_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_P4", &multilepton_JetHighestPt1_2ndPair_P4, &b_multilepton_JetHighestPt1_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_CSV", &multilepton_JetHighestPt1_2ndPair_CSV, &b_multilepton_JetHighestPt1_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_JEC_Up", &multilepton_JetHighestPt1_2ndPair_JEC_Up, &b_multilepton_JetHighestPt1_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_JEC_Down", &multilepton_JetHighestPt1_2ndPair_JEC_Down, &b_multilepton_JetHighestPt1_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_JER_Up", &multilepton_JetHighestPt1_2ndPair_JER_Up, &b_multilepton_JetHighestPt1_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt1_2ndPair_JER_Down", &multilepton_JetHighestPt1_2ndPair_JER_Down, &b_multilepton_JetHighestPt1_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_Id", &multilepton_JetHighestPt2_2ndPair_Id, &b_multilepton_JetHighestPt2_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_P4", &multilepton_JetHighestPt2_2ndPair_P4, &b_multilepton_JetHighestPt2_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_CSV", &multilepton_JetHighestPt2_2ndPair_CSV, &b_multilepton_JetHighestPt2_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_JEC_Up", &multilepton_JetHighestPt2_2ndPair_JEC_Up, &b_multilepton_JetHighestPt2_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_JEC_Down", &multilepton_JetHighestPt2_2ndPair_JEC_Down, &b_multilepton_JetHighestPt2_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_JER_Up", &multilepton_JetHighestPt2_2ndPair_JER_Up, &b_multilepton_JetHighestPt2_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetHighestPt2_2ndPair_JER_Down", &multilepton_JetHighestPt2_2ndPair_JER_Down, &b_multilepton_JetHighestPt2_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_Id", &multilepton_JetClosestMw1_2ndPair_Id, &b_multilepton_JetClosestMw1_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_P4", &multilepton_JetClosestMw1_2ndPair_P4, &b_multilepton_JetClosestMw1_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_CSV", &multilepton_JetClosestMw1_2ndPair_CSV, &b_multilepton_JetClosestMw1_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_JEC_Up", &multilepton_JetClosestMw1_2ndPair_JEC_Up, &b_multilepton_JetClosestMw1_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_JEC_Down", &multilepton_JetClosestMw1_2ndPair_JEC_Down, &b_multilepton_JetClosestMw1_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_JER_Up", &multilepton_JetClosestMw1_2ndPair_JER_Up, &b_multilepton_JetClosestMw1_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw1_2ndPair_JER_Down", &multilepton_JetClosestMw1_2ndPair_JER_Down, &b_multilepton_JetClosestMw1_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_Id", &multilepton_JetClosestMw2_2ndPair_Id, &b_multilepton_JetClosestMw2_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_P4", &multilepton_JetClosestMw2_2ndPair_P4, &b_multilepton_JetClosestMw2_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_CSV", &multilepton_JetClosestMw2_2ndPair_CSV, &b_multilepton_JetClosestMw2_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_JEC_Up", &multilepton_JetClosestMw2_2ndPair_JEC_Up, &b_multilepton_JetClosestMw2_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_JEC_Down", &multilepton_JetClosestMw2_2ndPair_JEC_Down, &b_multilepton_JetClosestMw2_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_JER_Up", &multilepton_JetClosestMw2_2ndPair_JER_Up, &b_multilepton_JetClosestMw2_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetClosestMw2_2ndPair_JER_Down", &multilepton_JetClosestMw2_2ndPair_JER_Down, &b_multilepton_JetClosestMw2_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_Id", &multilepton_JetLowestMjj1_2ndPair_Id, &b_multilepton_JetLowestMjj1_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_P4", &multilepton_JetLowestMjj1_2ndPair_P4, &b_multilepton_JetLowestMjj1_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_CSV", &multilepton_JetLowestMjj1_2ndPair_CSV, &b_multilepton_JetLowestMjj1_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_JEC_Up", &multilepton_JetLowestMjj1_2ndPair_JEC_Up, &b_multilepton_JetLowestMjj1_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_JEC_Down", &multilepton_JetLowestMjj1_2ndPair_JEC_Down, &b_multilepton_JetLowestMjj1_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_JER_Up", &multilepton_JetLowestMjj1_2ndPair_JER_Up, &b_multilepton_JetLowestMjj1_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj1_2ndPair_JER_Down", &multilepton_JetLowestMjj1_2ndPair_JER_Down, &b_multilepton_JetLowestMjj1_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_Id", &multilepton_JetLowestMjj2_2ndPair_Id, &b_multilepton_JetLowestMjj2_2ndPair_Id);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_P4", &multilepton_JetLowestMjj2_2ndPair_P4, &b_multilepton_JetLowestMjj2_2ndPair_P4);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_CSV", &multilepton_JetLowestMjj2_2ndPair_CSV, &b_multilepton_JetLowestMjj2_2ndPair_CSV);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_JEC_Up", &multilepton_JetLowestMjj2_2ndPair_JEC_Up, &b_multilepton_JetLowestMjj2_2ndPair_JEC_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_JEC_Down", &multilepton_JetLowestMjj2_2ndPair_JEC_Down, &b_multilepton_JetLowestMjj2_2ndPair_JEC_Down);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_JER_Up", &multilepton_JetLowestMjj2_2ndPair_JER_Up, &b_multilepton_JetLowestMjj2_2ndPair_JER_Up);
//   fChain->SetBranchAddress("multilepton_JetLowestMjj2_2ndPair_JER_Down", &multilepton_JetLowestMjj2_2ndPair_JER_Down, &b_multilepton_JetLowestMjj2_2ndPair_JER_Down);
//   fChain->SetBranchAddress("multilepton_h0_Id", &multilepton_h0_Id, &b_multilepton_h0_Id);
//   fChain->SetBranchAddress("multilepton_h0_P4", &multilepton_h0_P4, &b_multilepton_h0_P4);
//   fChain->SetBranchAddress("multilepton_h0_Label", &multilepton_h0_Label, &b_multilepton_h0_Label);
//   fChain->SetBranchAddress("multilepton_t1_Id", &multilepton_t1_Id, &b_multilepton_t1_Id);
//   fChain->SetBranchAddress("multilepton_t1_P4", &multilepton_t1_P4, &b_multilepton_t1_P4);
//   fChain->SetBranchAddress("multilepton_t1_Label", &multilepton_t1_Label, &b_multilepton_h0_Label);
//   fChain->SetBranchAddress("multilepton_t2_Id", &multilepton_t2_Id, &b_multilepton_t2_Id);
//   fChain->SetBranchAddress("multilepton_t2_P4", &multilepton_t2_P4, &b_multilepton_t2_P4);
//   fChain->SetBranchAddress("multilepton_t2_Label", &multilepton_t2_Label, &b_multilepton_h0_Label);
//   fChain->SetBranchAddress("multilepton_mET", &multilepton_mET, &b_multilepton_mET);
//   fChain->SetBranchAddress("multilepton_mETcov00", &multilepton_mETcov00, &b_multilepton_mETcov00);
//   fChain->SetBranchAddress("multilepton_mETcov01", &multilepton_mETcov01, &b_multilepton_mETcov01);
//   fChain->SetBranchAddress("multilepton_mETcov10", &multilepton_mETcov10, &b_multilepton_mETcov10);
//   fChain->SetBranchAddress("multilepton_mETcov11", &multilepton_mETcov11, &b_multilepton_mETcov11);
//   fChain->SetBranchAddress("multilepton_mHT", &multilepton_mHT, &b_multilepton_mHT);
//   fChain->SetBranchAddress("multilepton_Ptot", &multilepton_Ptot, &b_multilepton_Ptot);
   Notify();
}

Bool_t base::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void base::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t base::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

Double_t base::GetVal(TString varName){
	if(varName=="nJet25_Recl")return nJet25_Recl;
	if(varName=="max_Lep_eta")return max_Lep_eta;
	if(varName=="MT_met_lep1")return MT_met_lep1;
	if(varName=="mindr_lep1_jet")return mindr_lep1_jet;
	if(varName=="mindr_lep2_jet")return mindr_lep2_jet;
	if(varName=="LepGood_conePt0")return LepGood_conePt0;
	if(varName=="LepGood_conePt1")return LepGood_conePt1;
	if(varName=="mc_ttZhypAllowed")return mc_ttZhypAllowed;
	if(varName=="is_3l_TTH_SR")return is_3l_TTH_SR;
	if(varName=="weight")return weight;
	return 0;
}
#endif // #ifdef base_h
