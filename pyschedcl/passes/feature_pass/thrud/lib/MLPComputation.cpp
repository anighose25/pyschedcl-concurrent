#include "thrud/MLPComputation.h"

#include "thrud/DataTypes.h"
#include "thrud/MathUtils.h"
#include "thrud/Utils.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/PostDominators.h"

#include <algorithm>

//------------------------------------------------------------------------------
InstVector filterUsers(InstVector &insts, BasicBlock *block) {
  InstVector result;
  result.reserve(insts.size());
  std::copy_if(
      insts.begin(), insts.end(), std::back_inserter(result),
      [block](Instruction *inst) { return (inst->getParent() == block); });

  return result;
}

//------------------------------------------------------------------------------
bool isLoad(const llvm::Instruction &inst) { return isa<LoadInst>(inst); }

//------------------------------------------------------------------------------
int countLoadsBounded(Instruction *def, Instruction *user) {
  BasicBlock::iterator iter(def), end(user);
  ++iter;
  return std::count_if(iter, end, isLoad);
}

//------------------------------------------------------------------------------
int countLoads(BlockVector blocks, BasicBlock *defBlock, BasicBlock *userBlock,
               Instruction *def, Instruction *user) {

  if (defBlock == userBlock) {
    return countLoadsBounded(def, user);
  }

  int result = 0;
  for (auto block : blocks) {
    if (block == defBlock) {
      result += countLoadsBounded(def, block->end());
      continue;
    }
    if (block == userBlock) {
      result += countLoadsBounded(block->begin(), user);
      continue;
    }

    result += countLoadsBounded(block->begin(), block->end());
  }
  return result;
}

//------------------------------------------------------------------------------
// FIXME: this might not work with loops.
BlockVector getRegionBlocks(BasicBlock *defBlock, BasicBlock *userBlock) {
  BlockVector result;
  BlockStack stack;
  stack.push(userBlock);
  errs() <<"MLP Computation 68: Entering getRegionBlocks \n";
  while (!stack.empty()) {
    // Pop the first block.
    errs() << "MLP Computation 70: Getting stack top\n";
    BasicBlock *block = stack.top();
    result.push_back(block);
    errs()<<"MLP Computation 73: Popping first block\n";
    stack.pop();
    errs() <<"MLP Computation 75: Don't put defBlock Predecessors\n";
    // Don't put to the stack the defBlock predecessors.
    if (block == defBlock)
    {
        errs() << "MLP Computation 79 Entering after check \n";
        continue;
    }
     errs() <<"MLP Computation 82: Pushing predecessors of defBlock\n";
    // Push to the stack the defBlock predecessors.
    for (pred_iterator iter = pred_begin(block), end = pred_end(block);
         iter != end; ++iter) {
      BasicBlock *pred = *iter;
      stack.push(pred);
    }
  }

  return result;
}

//------------------------------------------------------------------------------
int computeDistance(Instruction *def, Instruction *user) {
  BasicBlock *defBlock = def->getParent();
  BasicBlock *userBlock = user->getParent();
  user->dump();
  errs() << "MLP Computation 94: Computing Distance\n";
  user->dump();
  // Manage the special case in which the user is a phi-node.
  if (PHINode *phi = dyn_cast<PHINode>(user)) {
      errs() <<"MLP Computation 99: Casting Phi Instruction \n";
      phi->dump();
      for (unsigned int index = 0; index < phi->getNumIncomingValues(); ++index) {
       errs() << "MLP Computation 101: Get Incoming Phi Blocks \n";
       if (def == phi->getIncomingValue(index)) {
        userBlock = phi->getIncomingBlock(index);
        BlockVector blocks = getRegionBlocks(defBlock, userBlock);
        return countLoads(blocks, defBlock, userBlock, def, userBlock->end());
      }
    }
  }

  BlockVector blocks = getRegionBlocks(defBlock, userBlock);
  return countLoads(blocks, defBlock, userBlock, def, user);
}

//------------------------------------------------------------------------------
// MLP computation.
// MLP: count the number of loads that fall in each load-use interval
// (interval between a load and the first use of the loaded value).
float getMLP(BasicBlock *block) {
  std::vector<int> distances;
  errs() <<"MLP Computation 116: Computing number of loads \n";
  for (auto inst = block->begin(), end = block->end(); inst != end; ++inst) {
     
    if (isa<LoadInst>(inst)) {
      inst->dump();
      InstVector users = findUsers(inst);
      users = filterUsers(users, block);
      errs() << "MLP Computation 116: Entering Transformation \n";
      std::transform(
          users.begin(), users.end(), distances.begin(),
          [inst](Instruction *user) { return computeDistance(inst, user); });
    }
  }
 errs() << "MLP Computation 119 Computing average\n" ;
 for(int i=0;i<distances.size();i++)
     errs()<<distances[i]<<" ";
    errs()<<"\n";
  return getAverage(distances);
}
