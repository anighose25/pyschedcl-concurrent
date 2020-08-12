#include "thrud/FeatureCollector.h"

#include "thrud/DataTypes.h"
#include "thrud/Graph.h"
#include "thrud/MathUtils.h"
#include "thrud/OCLEnv.h"
#include "thrud/Utils.h"
#include "thrud/SubscriptAnalysis.h"

#include "thrud/ILPComputation.h"
#include "thrud/MLPComputation.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/YAMLTraits.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>

using namespace llvm;

using yaml::MappingTraits;
using yaml::SequenceTraits;
using yaml::IO;
using yaml::Output;

namespace llvm {
namespace yaml {

//------------------------------------------------------------------------------
template <> struct MappingTraits<FeatureCollector> {
  static void mapping(IO &io, FeatureCollector &collector) {
    for (auto &iter : collector.instTypes) {
      io.mapRequired(iter.first.c_str(), iter.second);
    }

        // Instructions per block.
    io.mapRequired("instsPerBlock", collector.blockInsts);

    // Dump Phi nodes.
    std::vector<int> args;

    for (auto &iter : collector.blockPhis) {
      std::vector<std::string> phis = iter.second;
      int argSum = 0;
      for (auto &phiIter : phis) {
        int argNumber = collector.phiArgs[phiIter];
        argSum += argNumber;
      }
      args.push_back(argSum);
    }

    io.mapRequired("phiArgs", args);
    io.mapRequired("ilpPerBlock", collector.blockILP);
    io.mapRequired("mlpPerBlock", collector.blockMLP);
    io.mapRequired("avgLiveRange", collector.avgLiveRange);
    io.mapRequired("aliveOut", collector.aliveOutBlocks);
  }
};

//------------------------------------------------------------------------------
template <> struct MappingTraits<std::pair<float, float> > {
  static void mapping(IO &io, std::pair<float, float> &avgVar) {
    io.mapRequired("avg", avgVar.first);
    io.mapRequired("var", avgVar.second);
  }
};

//------------------------------------------------------------------------------
// Sequence of ints.
template <> struct SequenceTraits<std::vector<int> > {
  static size_t size(IO &, std::vector<int> &seq) { return seq.size(); }
  static int &element(IO &, std::vector<int> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};

//------------------------------------------------------------------------------
// Sequence of floats.
template <> struct SequenceTraits<std::vector<float> > {
  static size_t size(IO &, std::vector<float> &seq) { return seq.size(); }
  static float &element(IO &, std::vector<float> &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};

//------------------------------------------------------------------------------
// Sequence of pairs.
template <> struct SequenceTraits<std::vector<std::pair<float, float> > > {
  static size_t size(IO &, std::vector<std::pair<float, float> > &seq) {
    return seq.size();
  }
  static std::pair<float, float> &
  element(IO &, std::vector<std::pair<float, float> > &seq, size_t index) {
    if (index >= seq.size())
      seq.resize(index + 1);
    return seq[index];
  }

  static const bool flow = true;
};
}
}
//Support Functions
void getIndVar(Loop *loop);
void findUsesOf(Instruction *inst, InstSet &result);
std::string getOpcodeSymbol(std::string opcode);
std::string getBinaryOpString(BinaryOperator *B, std::map<Value*, std::string>& Value_to_Str_Map);
void processIndVar(std::map<Value*,std::tuple<Value*,int,int> >& IndVarMap, Loop* loop);
bool isKeyinValueMap(std::map<Value *,std::string> ValueMap, Value *key);
//------------------------------------------------------------------------------
FeatureCollector::FeatureCollector() {
// Instruction-specific counters.
#define HANDLE_INST(N, OPCODE, CLASS) instTypes[#OPCODE] = 0;
#include "llvm/IR/Instruction.def"

  // Initialize all counters.
  instTypes["insts"] = 0;
  instTypes["blocks"] = 0;
  instTypes["edges"] = 0;
  instTypes["criticalEdges"] = 0;
  instTypes["condBranches"] = 0;
  instTypes["uncondBranches"] = 0;
  instTypes["fourB"] = 0;
  instTypes["eightB"] = 0;
  instTypes["fps"] = 0;
  instTypes["vector2"] = 0;
  instTypes["vector4"] = 0;
  instTypes["vectorOperands"] = 0;
  instTypes["localLoads"] = 0;
  instTypes["localStores"] = 0;
  instTypes["mathFunctions"] = 0;
  instTypes["barriers"] = 0;
  instTypes["args"] = 0;
  instTypes["divRegions"] = 0;
  instTypes["divInsts"] = 0;
  instTypes["divRegionInsts"] = 0;
  instTypes["uniformLoads"] = 0;
}

//------------------------------------------------------------------------------
void FeatureCollector::computeILP(BasicBlock *block) {
  blockILP.push_back(getILP(block));
}

//------------------------------------------------------------------------------
void FeatureCollector::computeMLP(BasicBlock *block) {
  blockMLP.push_back(getMLP(block));
}

//------------------------------------------------------------------------------
void FeatureCollector::countIncomingEdges(const BasicBlock &block) {
  const BasicBlock *tmpBlock = (const BasicBlock *)&block;
  const_pred_iterator first = pred_begin(tmpBlock), last = pred_end(tmpBlock);
  blockIncoming[block.getName()] = std::distance(first, last);
}

//------------------------------------------------------------------------------
void FeatureCollector::countOutgoingEdges(const BasicBlock &block) {
  blockOutgoing[block.getName()] = block.getTerminator()->getNumSuccessors();
}

//------------------------------------------------------------------------------
void FeatureCollector::countInstsBlock(const BasicBlock &block) {
  blockInsts.push_back(static_cast<int>(block.getInstList().size()));
}

//------------------------------------------------------------------------------
void FeatureCollector::countEdges(const Function &function) {
  int criticalEdges = 0;

  int edges = std::accumulate(function.begin(), function.end(), 0,
                              [](int result, const BasicBlock &block) {
    return result + block.getTerminator()->getNumSuccessors();
  });

  for (auto &block : function) {
    const TerminatorInst *termInst = block.getTerminator();
    int termNumber = termInst->getNumSuccessors();

    for (int index = 0; index < termNumber; ++index) {
      criticalEdges += isCriticalEdge(termInst, index);
    }
  }

  instTypes["edges"] = edges;
  instTypes["criticalEdges"] = criticalEdges;
}

//------------------------------------------------------------------------------
void FeatureCollector::countBranches(const Function &function) {
  int condBranches = 0;
  int uncondBranches = 0;
  for (auto &block : function) {
    const TerminatorInst *term = block.getTerminator();
    if (const BranchInst *branch = dyn_cast<BranchInst>(term)) {
      if (branch->isConditional())
        ++condBranches;
      else
        ++uncondBranches;
    }
  }

  instTypes["condBranches"] = condBranches;
  instTypes["uncondBranches"] = uncondBranches;
}

void FeatureCollector::getFunctionSignature(const Function &function) {
  int index = 0;
  std::map<std::string,int> arrayNameToFuncIDmap;
  int arg_counter=0;
  for (auto &A : function.getArgumentList()){
      const Value* A_value = cast<Value>(&A);
      std::string A_name = A_value->getName().str();
      arrayNameToFuncIDmap[A_name]=arg_counter++;
  }
  for (auto &A : function.getArgumentList())
  {
      //errs()<<index<< "\n";
      //A.dump();
      Type *ty = A.getType();
      if (ty->isPointerTy())
      {     
          Type *ety = ty->getPointerElementType();
          if(ety->isFloatTy())
              argIndexElementTypeMap[index] = "float"; 
          if(ety->isIntegerTy())
              argIndexElementTypeMap[index] = "int";
          //InstVector users = findUsersOfArguments(&A); 
          bool buffer_input_type = false;
          bool buffer_output_type = false;
          for(auto start = inst_begin(function), end = inst_end(function);start!=end;++start)
          {
              if(const LoadInst *L = dyn_cast<LoadInst>(& *start))
              {
          //        errs()<<"Feature Collector 241: Found load instruction\n";
                  const Value *v=L->getPointerOperand();
                  const GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(&(*v));
                  Value *firstOperand  = gep->getOperand(0);
            //      errs()<<"Feature Collector 245: Dumping array load\n";
                  //firstOperand->dump();
                  const Value * A_value = cast<Value>(&A);
                  bool equal = (A_value == firstOperand);
                  //errs()<<equal<<"\n";
                  if(equal)
                    buffer_input_type=true;
                      

              }
              if(const StoreInst *L = dyn_cast<StoreInst>(& *start))
              {
                  //errs()<<"Feature Collector 241: Found load instruction\n";
                  const Value *v=L->getPointerOperand();
                  const GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(&(*v));
                  Value *firstOperand  = gep->getOperand(0);
                  //errs()<<"Feature Collector 245: Dumping array load\n";
                  //firstOperand->dump();
                  const Value * A_value = cast<Value>(&A);
                  bool equal = (A_value == firstOperand);
                  if(equal)
                      buffer_output_type= true;

              }

          }
          if(buffer_input_type && buffer_output_type)
              argIndexElementTypeMap[index]+="_io";
          else if(buffer_output_type)
              argIndexElementTypeMap[index]+="_output";
          else if(buffer_input_type)
              argIndexElementTypeMap[index]+="_input";
          else
              argIndexElementTypeMap[index]+="_input";
      }

      else if(ty->isIntegerTy())
          argIndexElementTypeMap[index] = "int";
      else if (ty->isFloatTy())
          argIndexElementTypeMap[index] = "float";



      index ++;
  }
// for (auto iter : argIndexElementTypeMap)
//     errs() << iter.first << " -> " << iter.second << "\n";
  for(unsigned int i=0;i<argIndexElementTypeMap.size();i++)
    errs()<<i<<" -> "<<argIndexElementTypeMap[i]<<","<<argIndexElementDimMap[i].get_dataset()<<"\n"; 
  for(auto iter : arrayAccessExpressions){
    errs()<<arrayNameToFuncIDmap[iter.first]<<":";
    for(unsigned int j=0; j<iter.second.size();j++){
      errs()<<iter.second[j];
      if(j!=iter.second.size()-1)
        errs()<<",";
      else
        errs()<<"\n";
    }

  }
  

}

void findUsesOfInst(Instruction *inst, InstSet &result) {
  
  for (auto userIter = inst->user_begin(), userEnd = inst->user_end();
       userIter != userEnd; ++userIter) {
    if (Instruction *userInst = dyn_cast<Instruction>(*userIter)) {
      result.insert(userInst);
    }
  }
}

bool isKeyinValueMap(std::map<Value *,std::string> ValueMap, Value *key)
{
  if (ValueMap.find(key) == ValueMap.end())
    return false;
  else
    return true;
}

void getIndVar(Loop *loop)
{
  BasicBlock *H = loop->getHeader();
  BasicBlock *L = loop->getLoopLatch();

  
   

  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ++I) 
  {
    PHINode *PN = cast<PHINode>(I);
    LOG_ACCESS errs()<<"Possible induction variable?\n";
    LOG_ACCESS PN->dump();
    LOG_ACCESS errs()<<"Initial Value and Increment Operation?\n";
    unsigned IncomingEdge = loop->contains(PN->getIncomingBlock(0));
    unsigned BackEdge     = IncomingEdge^1;
    // Check incoming value.
    Value *InitValueVal = PN->getIncomingValue(IncomingEdge);
    Value *Incr = PN->getIncomingValue(BackEdge);
    LOG_ACCESS InitValueVal->dump();
    LOG_ACCESS Incr->dump();
    LOG_ACCESS errs()<<"Users of induction variable\n";
    LOG_ACCESS for (auto userIter = I->user_begin(), userEnd = I->user_end();
       userIter != userEnd; ++userIter) {
    if (Instruction *userInst = dyn_cast<Instruction>(*userIter)) {
      userInst->dump();
    }
  }

  }
  LOG_ACCESS errs()<<"Latch terminator branch condition \n";
  Instruction *terminator = L->getTerminator();
  LOG_ACCESS if (BranchInst *BI = dyn_cast<BranchInst>(terminator))
  if (BI->isConditional()) 
    if(CmpInst *compare = dyn_cast<CmpInst>(BI->getCondition())){
      Value *Op0 = compare->getOperand(0);
      Value *Op1 = compare->getOperand(1);
      Op0->dump();
      Op1->dump();
      
    }

    
    
    
  
  

}
std::string getOpcodeSymbol(std::string opcode)
{
  
  if(opcode == std::string("mul"))
    return std::string("*");
  else if(opcode == std::string("add"))
    return std::string("+");
  else 
    return std::string("ERR");
  

}

std::string getBinaryOpString(BinaryOperator *B, std::map<Value*, std::string>& Value_to_Str_Map)
{
  Value *op1 = B->getOperand(0);
  Value *op2 = B->getOperand(1);
  
  std::string op1_name;
  std::string op2_name;
  std::string opcode = std::string(B->getOpcodeName());
  
  if (const ConstantInt *constInt = dyn_cast<ConstantInt>(op1))
  {
    std::stringstream ss;
    ss << constInt->getSExtValue();
    
    op1_name = ss.str(); 
  } 
  else if(isKeyinValueMap(Value_to_Str_Map,op1)){

    op1_name = Value_to_Str_Map[op1];
    LOG_ACCESS errs()<<"Operand 1 in Map: "<<op1_name<<"\n";
  }
  else{
    op1_name = op1->getName().str();
    LOG_ACCESS errs()<<"Operand 1 not in Map: "<<op1_name<<"\n";
  }
  
  if (const ConstantInt *constInt = dyn_cast<ConstantInt>(op2))
  {
    std::stringstream ss;
    ss << constInt->getSExtValue();
    op2_name = ss.str(); 
  } 
  else if(isKeyinValueMap(Value_to_Str_Map,op2)){

    op2_name = Value_to_Str_Map[op2];
    LOG_ACCESS errs()<<"Operand 2 in Map: "<<op2_name<<"\n";
  }
  else
  {
    op2_name = op2->getName().str();
    LOG_ACCESS errs()<<"Operand 2 not in Map: "<<op2_name<<"\n";
  }
 
  LOG_ACCESS errs()<<"Operands: "<<op1_name<<"|"<<op2_name<<"\n";
  std::string opcode_sym = getOpcodeSymbol(opcode);
  std::string result= "("+op1_name + " " + opcode_sym + " " + op2_name+")";  
  
  return result;
}



void processIndVar(std::map<Value*,std::tuple<Value*,int,int> >& IndVarMap, Loop* loop)
{
      while (true) {
          std::map<Value*, std::tuple<Value*, int, int> > NewMap = IndVarMap;
          auto blocks_in_loop = loop->getBlocks();
          for (auto B: blocks_in_loop) {
          
            for (auto &I : *B) {
              // we only accept multiplication, addition, and subtraction
              // we only accept constant integer as one of theoperands
              if (auto *op = dyn_cast<BinaryOperator>(&I)) {
                Value *lhs = op->getOperand(0);
                Value *rhs = op->getOperand(1);
                // check if one of the operands belongs to indvars
                if (IndVarMap.count(lhs) || IndVarMap.count(rhs)) {
                  // case: Add
                  if (I.getOpcode() == Instruction::Add) {
                    ConstantInt* CIL = dyn_cast<ConstantInt>(lhs);
                    ConstantInt* CIR = dyn_cast<ConstantInt>(rhs);
                    if (IndVarMap.count(lhs) && CIR) {
                      std::tuple<Value*, int, int> t = IndVarMap[lhs];
                      int new_val = CIR->getSExtValue() + std::get<2>(t);
                      NewMap[&I] = std::make_tuple(std::get<0>(t), std::get<1>(t), new_val);
                    } else if (IndVarMap.count(rhs) && CIL) {
                      std::tuple<Value*, int, int> t = IndVarMap[rhs];
                      int new_val = CIL->getSExtValue() + std::get<2>(t);
                      NewMap[&I] = std::make_tuple(std::get<0>(t), std::get<1>(t), new_val);
                    }
                  // case: Sub
                  } else if (I.getOpcode() == Instruction::Sub) {
                    ConstantInt* CIL = dyn_cast<ConstantInt>(lhs);
                    ConstantInt* CIR = dyn_cast<ConstantInt>(rhs);
                    if (IndVarMap.count(lhs) && CIR) {
                      std::tuple<Value*, int, int> t = IndVarMap[lhs];
                      int new_val = std::get<2>(t) - CIR->getSExtValue();
                      NewMap[&I] = std::make_tuple(std::get<0>(t), std::get<1>(t), new_val);
                    } else if (IndVarMap.count(rhs) && CIL) {
                      std::tuple<Value*, int, int> t = IndVarMap[rhs];
                      int new_val = std::get<2>(t) - CIL->getSExtValue();
                      NewMap[&I] = std::make_tuple(std::get<0>(t), std::get<1>(t), new_val);
                    }
                  // case: Mul
                  } else if (I.getOpcode() == Instruction::Mul) {
                    ConstantInt* CIL = dyn_cast<ConstantInt>(lhs);
                    ConstantInt* CIR = dyn_cast<ConstantInt>(rhs);
                    if (IndVarMap.count(lhs) && CIR) {
                      std::tuple<Value*, int, int> t = IndVarMap[lhs];
                      int new_val = CIR->getSExtValue() * std::get<1>(t);
                      NewMap[&I] = std::make_tuple(std::get<0>(t), new_val, std::get<2>(t));
                    } else if (IndVarMap.count(rhs) && CIL) {
                      std::tuple<Value*, int, int> t = IndVarMap[rhs];
                      int new_val = CIL->getSExtValue() * std::get<1>(t);
                      NewMap[&I] = std::make_tuple(std::get<0>(t), new_val, std::get<2>(t));
                    }
                  }
                } // if operand in indvar
              } // if op is binop
            } // auto &I: B
          } // auto &B: blks
          if (NewMap.size() == IndVarMap.size()) break;
          else IndVarMap = NewMap;
        }

}


void FeatureCollector::getAccessExpressions(Function &function, NDRange *ndr, LoopInfo *loopInfo, PostDominatorTree *pdt) {

  // get tid expressions 

   InstVector tids = ndr->getTidInformation();
   std::map<Value*, std::string> Value_to_Str_Map;

   for (InstVector::iterator iter = tids.begin(), iterEnd = tids.end();
       iter != iterEnd; ++iter){
   
      Instruction *inst = *iter;    
      int direction = ndr->getDirection(inst);
      Value *registerName = cast<Value>(inst);
      std::string regname = registerName->getName().str();
      std::stringstream ss; 
      if(ndr->isGlobal(inst)){
        ss<<"global_id("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }
      
      else if(ndr->isLocal(inst)){
        ss<<"local_id("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }
      
      else if(ndr->isLocalSize(inst)){
        ss<<"local_size("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }
      
      else if(ndr->isGlobalSize(inst)){
        ss<<"global_size("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }
      
      else if(ndr->isGroupId(inst)){
        ss<<"group_id("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }

      else if(ndr->isGroupsNum(inst)){
        ss<<regname<<"group_num("<<direction<<")";   
        LOG_ACCESS errs()<<regname<<ss.str()<<"\n";
      }
      Value_to_Str_Map[registerName]=ss.str();    
  
  }

  //2. Get Function Argument Values

  for (auto &A : function.getArgumentList()){
      Value * A_value = cast<Value>(&A);
      std::string A_name = A_value->getName().str();
      Value_to_Str_Map[A_value]=A_name;
      LOG_ACCESS errs()<<A_name<<"\n";
   }  



  

  //3. Get BinaryInst Values


  


  for(auto start = inst_begin(function), end = inst_end(function);start!=end;++start)
  {
  
    if(BinaryOperator *B = dyn_cast<BinaryOperator>(& *start))
    {

      Value *registerName = cast<Value>(& *start);
      if(!isKeyinValueMap(Value_to_Str_Map,registerName))
      {
        std::string regName = registerName->getName().str();
        std::string result = getBinaryOpString(B,Value_to_Str_Map);
        if(regName.length())
          Value_to_Str_Map[registerName]=result;
        LOG_ACCESS registerName->dump();
        LOG_ACCESS errs()<<regName<<" "<<getBinaryOpString(B,Value_to_Str_Map)<<"\n";
      }
    }    
  
  }
  //4. Get ArrayInst Values
  LOG_ACCESS errs()<<"==========================================\n";  

  for(auto start = inst_begin(function), end = inst_end(function);start!=end;++start)
  {
      if(const LoadInst *L = dyn_cast<LoadInst>(& *start))
      {
          LOG_ACCESS errs()<<"Found load instruction\n";
          const Value *v=L->getPointerOperand();
          const GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(&(*v));
          Value *firstOperand  = gep->getOperand(0);
          Value *secondOperand  = gep->getOperand(1);
          LOG_ACCESS firstOperand->dump();
          LOG_ACCESS errs()<<"Access Expression: "<<Value_to_Str_Map[secondOperand]<<"\n";
          arrayAccessExpressions[firstOperand->getName().str()].push_back(Value_to_Str_Map[secondOperand]);
      }
      if(const StoreInst *L = dyn_cast<StoreInst>(& *start))
      {
          LOG_ACCESS errs()<<"Found store instruction\n";
          const Value *v=L->getPointerOperand();
          const GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(&(*v));
          Value *firstOperand  = gep->getOperand(0);
          Value *secondOperand  = gep->getOperand(1);
          LOG_ACCESS firstOperand->dump();
          LOG_ACCESS errs()<<"Access Expression: "<<Value_to_Str_Map[secondOperand]<<"\n";
          arrayAccessExpressions[firstOperand->getName().str()].push_back(Value_to_Str_Map[secondOperand]);
      }

  }
  

  
  LOG_ACCESS errs()<<"==========================================\n";
  
   //3. Get PhiNode and Induction Variable Values
  // Manage branches.
   
  // for(auto start = inst_begin(function), end = inst_end(function);start!=end;++start){
  //   if (BranchInst *branch = dyn_cast<BranchInst>(& *start)) {
        
  //       BasicBlock *header = branch->getParent();
  //       BasicBlock *exiting = findImmediatePostDom(header, pdt);
  
  //       if (loopInfo->isLoopHeader(header)) {
  //          Loop *loop = loopInfo->getLoopFor(header);
  //          if (loop == loopInfo->getLoopFor(exiting))
  //            exiting = loop->getExitBlock();
          
  //       for (auto &I : *header) {
  //         if (PHINode *PN = dyn_cast<PHINode>(&I)) {
  //           errs()<<"Possible Induction Variable: ";
  //           PN->dump();
  //         }
  //       }
  //       auto blocks_in_loop =loop->getBlocks();
  //       std::map<Value*, std::tuple<Value*, int, int> > IndVarMap;
  //       processIndVar(IndVarMap,loop);

  //     }
  //   }
  // }

  for (auto loop_start = loopInfo->begin(), loop_end = loopInfo->end(); loop_start != loop_end; ++loop_start) {

    
    Loop *current_loop= *loop_start;
    LOG_ACCESS errs()<<" Loop Depth: "<<current_loop->getLoopDepth()<<" Simplified?  "<<current_loop->isLoopSimplifyForm()<<"\n";
    getIndVar(current_loop);

    LOG_ACCESS errs()<<" Searching for subloops\n"; 
    const std::vector<Loop *> loopVector = current_loop->getSubLoops();
    if(loopVector.size()>0)
      LOG_ACCESS errs()<<"Found\n";
    else
      LOG_ACCESS errs()<<"Not Found\n";
    for(unsigned int i=0; i<loopVector.size();i++)
    {
      LOG_ACCESS errs()<<" Loop Depth: "<<loopVector[i]->getLoopDepth()<<" Simplified?  "<<loopVector[i]->isLoopSimplifyForm()<<"\n";
      getIndVar(loopVector[i]);
    }
    
  } 


  // for (auto iter : Value_to_Str_Map){
  //   (iter.first)->dump();
  //   errs() << iter.second << "\n";
  // }
  // errs()<<"==========================================\n";
}
void FeatureCollector::getFunctionTids(Function &function, NDRange *ndr) {
   std::vector<Value*>funcArgs;
   int counter=0;
   for (auto &A : function.getArgumentList())
   {
      Value * A_value = cast<Value>(&A);
      funcArgs.push_back(A_value);
      argIndexElementDimMap[counter]=*(new dim3);
      counter++;
   }
   InstVector tids = ndr->getTids();
   for (InstVector::iterator iter = tids.begin(), iterEnd = tids.end();
       iter != iterEnd; ++iter) {
  
    std::vector<User *> users;
    std::vector<Instruction*> seeds;
    Instruction *inst = *iter;
    LOG_DEBUG errs()<<"\n=====================================================\n";
    int direction = ndr->getDirection(inst);
    LOG_DEBUG errs()<<"Direction: "<<direction<<"\n";
    
    LOG_DEBUG inst->dump();
    std::copy(inst->user_begin(), inst->user_end(), std::back_inserter(users));
    LOG_DEBUG errs()<<"Seed instructions:\n";
    for(unsigned int i=0;i<users.size();i++) {
      Instruction *used_inst = dyn_cast<Instruction>(users[i]);
      LOG_DEBUG used_inst->dump();
      seeds.push_back(used_inst);
    
    }
  
    InstSet worklist(seeds.begin(), seeds.end());
    int print_counter = 0;
    while (!worklist.empty()) {
      auto iter = worklist.begin();
      Instruction *inst = *iter;
      LOG_DEBUG errs()<<"Iteration: "<<print_counter<<" Current Instruction\n";
      LOG_DEBUG inst->dump();
      worklist.erase(iter);
      std::vector<Instruction*> processedInsts;
      processedInsts.push_back(inst);
      InstSet users_of_inst;
      if(GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(inst))
      {
        Value *firstOperand  = gep->getOperand(0);
        LOG_DEBUG errs()<<"Array instruction found\n";
        LOG_DEBUG firstOperand->dump();
        for(unsigned int i=0;i<funcArgs.size();i++)
        {
          bool equal = firstOperand == funcArgs[i];
          if(equal)
          {
            LOG_DEBUG errs()<<"Arugment index: "<<i<<"\n";
            if(direction==0)
              argIndexElementDimMap[i].x=true;
            if(direction==1)
              argIndexElementDimMap[i].y=true;
            if(direction==2)
              argIndexElementDimMap[i].z=true;

          }

        
        }
      }
      findUsesOfInst(inst, users_of_inst);    
      LOG_DEBUG errs()<<"Used Instructions\n";  
      for (InstSet::iterator it = users_of_inst.begin(), iterEnd = users_of_inst.end(); it != iterEnd; ++it){
        Instruction *use_instance = *it;
        
        if(isa<BinaryOperator>(*it))
          if (!isPresent(*it, processedInsts))
          {
            LOG_DEBUG use_instance->dump();
            worklist.insert(*it);
            Value *registerName = cast<Value>(*it);
            std::string regName = registerName->getName().str();
            LOG_DEBUG errs()<<regName<<"\n";
          }
        if(GetElementPtrInst *gep=dyn_cast<GetElementPtrInst>(use_instance))
        {
          Value *firstOperand  = gep->getOperand(0);
          LOG_DEBUG errs()<<"Array instruction found\n";
          LOG_DEBUG firstOperand->dump();
          for(unsigned int i=0;i<funcArgs.size();i++)
          {
            bool equal = firstOperand == funcArgs[i];
            if(equal)
            {
              LOG_DEBUG errs()<<"Arugment index: "<<i<<"\n";
              if(direction==0)
                argIndexElementDimMap[i].x=true;
              if(direction==1)
                argIndexElementDimMap[i].y=true;
              if(direction==2)
                argIndexElementDimMap[i].z=true;

            }

          
          }
        }
     
      }
      print_counter++;
    if (print_counter==100)
      break;
    }
  }
  LOG_DEBUG
  for(unsigned int i=0;i<funcArgs.size();i++)
  {
    errs()<<"Argument "<<i<<" "<<argIndexElementDimMap[i].x<<" "<<argIndexElementDimMap[i].y<<" "<<argIndexElementDimMap[i].z<<"\n";
  }

}
//------------------------------------------------------------------------------
void FeatureCollector::countPhis(const BasicBlock &block) {
  std::vector<std::string> names;
  for (auto inst = block.begin(); isa<PHINode>(inst); ++inst) {
    std::string name = inst->getName();
    int argCount = inst->getNumOperands();

    phiArgs[name] = argCount;
    names.push_back(name);
  }

  blockPhis[block.getName()] = names;
}

//------------------------------------------------------------------------------
void FeatureCollector::countConstants(const BasicBlock &block) {
  int fourB = instTypes["fourB"];
  int eightB = instTypes["eightB"];
  int fps = instTypes["fps"];

  for (const auto &inst : block) {
    for (Instruction::const_op_iterator opIter = inst.op_begin(),
                                        opEnd = inst.op_end();
         opIter != opEnd; ++opIter) {
      const Value *operand = opIter->get();
      if (const ConstantInt *constInt = dyn_cast<ConstantInt>(operand)) {
        if (constInt->getBitWidth() == 32)
          ++fourB;

        if (constInt->getBitWidth() == 64)
          ++eightB;
      }

      if (isa<ConstantFP>(operand))
        ++fps;
    }
  }

  instTypes["fourB"] = fourB;
  instTypes["eightB"] = eightB;
  instTypes["fps"] = fps;
}

//------------------------------------------------------------------------------
void FeatureCollector::countBarriers(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const CallInst *callInst = dyn_cast<CallInst>(&inst)) {
      const Function *function = callInst->getCalledFunction();
      if (function == nullptr)
        continue;
      if (function->getName() == "barrier") {
        safeIncrement(instTypes, "barriers");
      }
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countMathFunctions(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const CallInst *callInst = dyn_cast<CallInst>(&inst)) {
      const Function *function = callInst->getCalledFunction();
      if (function == nullptr)
        continue;
      if (isMathName(function->getName())) {
        safeIncrement(instTypes, "mathFunctions");
      }
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countLocalMemoryUsage(const BasicBlock &block) {
  for (const auto &inst : block) {
    if (const LoadInst *loadInst = dyn_cast<LoadInst>(&inst)) {
      if (loadInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS)
        safeIncrement(instTypes, "localLoads");
    }
    if (const StoreInst *storeInst = dyn_cast<StoreInst>(&inst)) {
      if (storeInst->getPointerAddressSpace() == OCLEnv::LOCAL_AS)
        safeIncrement(instTypes, "localStores");
    }
  }
}

//------------------------------------------------------------------------------
void FeatureCollector::countDivInsts(Function &function,
                                     MultiDimDivAnalysis *mdda,
                                     SingleDimDivAnalysis *sdda) {
  // Insts in divergent regions.
  RegionVector &regions = mdda->getDivRegions();
  int divRegionInsts = std::accumulate(regions.begin(), regions.end(), 0,
                                       [](int result, DivergentRegion *region) {
    return result + region->size();
  });

  // Count uniform loads.
  int uniformLoads = 0;
  for (inst_iterator iter = inst_begin(function), iterEnd = inst_end(function);
       iter != iterEnd; ++iter) {
    Instruction *inst = &*iter;
    uniformLoads += isa<LoadInst>(inst) * !sdda->isDivergent(inst);
  }

  instTypes["divRegionInsts"] = divRegionInsts;
  instTypes["uniformLoads"] = uniformLoads;
  instTypes["divRegions"] = mdda->getDivRegions().size();
  instTypes["divInsts"] = mdda->getDivInsts().size();
}

//------------------------------------------------------------------------------
void FeatureCollector::countArgs(const Function &function) {
  instTypes["args"] = function.arg_size();
}

//------------------------------------------------------------------------------
int computeLiveRange(Instruction *inst) {
  Instruction *lastUser = findLastUser(inst);

  if (lastUser == nullptr)
    return inst->getParent()->size();

  assert(lastUser->getParent() == inst->getParent() &&
         "Different basic blocks");

  BasicBlock::iterator begin(inst), end(lastUser);

  return std::distance(begin, end);
}

//------------------------------------------------------------------------------
void FeatureCollector::livenessAnalysis(BasicBlock &block) {
  int aliveValues = 0;
  std::vector<int> ranges;

  for (auto &inst : block) {
    if (!inst.hasName())
      continue;

    bool isUsedElseWhere = isUsedOutsideOfDefiningBlock(&inst);
    aliveValues += isUsedElseWhere;

    if (!isUsedElseWhere) {
      int liveRange = computeLiveRange(&inst);
      ranges.push_back(liveRange);
    }
  }

  avgLiveRange.push_back(getAverage(ranges));
  aliveOutBlocks.push_back(aliveValues);
}

////------------------------------------------------------------------------------
//void FeatureCollector::countDimensions(NDRange *NDR) {
//  InstVector dir0 = NDR->getTids(0);
//  InstVector dir1 = NDR->getTids(1);
//  InstVector dir2 = NDR->getTids(2);
//
//  int dimensionNumber =
//      (dir0.size() != 0) + (dir1.size() != 0) + (dir2.size() != 0);
//
//  instTypes["dimensions"] = dimensionNumber;
//}

//------------------------------------------------------------------------------
void FeatureCollector::dump() {
  //errs() << "Feature Collector 389: Initializing YAML output stream\n"; 
  Output yout(errs());
  //errs() << "Feature Collector 391: Dumping Feature Collector Object \n";
  yout << *this;
}

//------------------------------------------------------------------------------
void FeatureCollector::loopCountEdges(const Function &function, LoopInfo *LI) {
  int edges = std::accumulate(function.begin(), function.end(), 0,
                              [LI](int result, const BasicBlock &block) {
    return result +
           (isInLoop(&block, LI) * block.getTerminator()->getNumSuccessors());
  });

  int criticalEdges = 0;
  for (auto &block : function) {
    if (!isInLoop(&block, LI))
      continue;

    const TerminatorInst *termInst = block.getTerminator();
    int termNumber = termInst->getNumSuccessors();

    for (int index = 0; index < termNumber; ++index) {
      criticalEdges += isCriticalEdge(termInst, index);
    }
  }

  instTypes["edges"] = edges;
  instTypes["criticalEdges"] = criticalEdges;
}

//------------------------------------------------------------------------------
void FeatureCollector::loopCountBranches(const Function &function,
                                         LoopInfo *LI) {
  int condBranches = 0;
  int uncondBranches = 0;
  for (auto &block : function) {

    if (!isInLoop(&block, LI))
      continue;

    const TerminatorInst *term = block.getTerminator();
    if (const BranchInst *branch = dyn_cast<BranchInst>(term)) {
      if (branch->isConditional() == true)
        ++condBranches;
      else
        ++uncondBranches;
    }
  }

  instTypes["condBranches"] = condBranches;
  instTypes["uncondBranches"] = uncondBranches;
}
//------------------------------------------------------------------------------
void FeatureCollector::loopCountDivInsts(Function &function,
                                         MultiDimDivAnalysis *mdda,
                                         SingleDimDivAnalysis *sdda,
                                         LoopInfo *LI) {
  // Count divergent regions.
  RegionVector &regions = mdda->getDivRegions();
  InstVector divInsts = mdda->getDivInsts();
  
  // Count number of divergent instructions.
  int divInstsCounter = std::accumulate(divInsts.begin(), divInsts.end(), 0,
                                    [LI](int result, Instruction *inst) {
    return result + (isInLoop(inst, LI));
  });

  // Count number of divergent regions.
  int divRegionsCounter = std::count_if(regions.begin(), regions.end(),
                                        [LI](DivergentRegion *region) {
    return isInLoop(region->getHeader(), LI);
  });

  // Count number of instructions in divergent regions.
  int divRegionInsts =
      std::accumulate(regions.begin(), regions.end(), 0,
                      [LI](int result, DivergentRegion *region) {
        return (result + (region->size() * isInLoop(region->getHeader(), LI)));
      });

  // Count uniform loads.
  int uniformLoads = 0;
  for (inst_iterator iter = inst_begin(function), iterEnd = inst_end(function);
       iter != iterEnd; ++iter) {
    Instruction *inst = &*iter;
    uniformLoads += isInLoop(inst, LI) * isa<LoadInst>(inst) * !sdda->isDivergent(inst);
  }

  instTypes["divInsts"] = divInstsCounter;
  instTypes["divRegions"] = divRegionsCounter;
  instTypes["divRegionInsts"] = divRegionInsts;
  instTypes["uniformLoads"] = uniformLoads;
}
