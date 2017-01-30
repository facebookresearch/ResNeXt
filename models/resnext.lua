--
--  Copyright (c) 2017, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNeXt model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end
   
   -- The original bottleneck residual layer
   local function resnet_bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
   
   -- The aggregated residual transformation bottleneck layer, Form (B)
   local function split(nInputPlane, d, c, stride)
      local cat = nn.ConcatTable()
      for i=1,c do
         local s = nn.Sequential()
         s:add(Convolution(nInputPlane,d,1,1,1,1,0,0))
         s:add(SBatchNorm(d))
         s:add(ReLU(true))
         s:add(Convolution(d,d,3,3,stride,stride,1,1))
         s:add(SBatchNorm(d))
         s:add(ReLU(true))
         cat:add(s)
      end
      return cat
   end
   
   local function resnext_bottleneck_B(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (opt.baseWidth/64))
      local C = opt.cardinality

      local s = nn.Sequential()
      s:add(split(nInputPlane, D, C, stride))
      s:add(nn.JoinTable(2))
      s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
    
   -- The aggregated residual transformation bottleneck layer, Form (C)
   local function resnext_bottleneck_C(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (opt.baseWidth/64))
      local C = opt.cardinality

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,D*C,1,1,1,1,0,0))
      s:add(SBatchNorm(D*C))
      s:add(ReLU(true))
      s:add(Convolution(D*C,D*C,3,3,stride,stride,1,1,C))
      s:add(SBatchNorm(D*C))
      s:add(ReLU(true))
      s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   local bottleneck
   if opt.bottleneckType == 'resnet' then 
      bottleneck = resnet_bottleneck
      print('Deploying ResNet bottleneck block')
   elseif opt.bottleneckType == 'resnext_B' then 
      bottleneck = resnext_bottleneck_B
      print('Deploying ResNeXt bottleneck block form B')
   elseif opt.bottleneckType == 'resnext_C' then 
      bottleneck = resnext_bottleneck_C
      print('Deploying ResNeXt bottleneck block form C (group convolution)')
   else
      error('invalid bottleneck type: ' .. opt.bottleneckType)
   end
   
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))
   
   elseif opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
      assert((depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101')
      local n = (depth - 2) / 9
      
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ' .. opt.dataset)

      model:add(Convolution(3,64,3,3,1,1,1,1))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(layer(bottleneck, 64, n, 1))
      model:add(layer(bottleneck, 128, n, 2))
      model:add(layer(bottleneck, 256, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(1024):setNumInputDims(3))
      local nCategories = opt.dataset == 'cifar10' and 10 or 100
      model:add(nn.Linear(1024, nCategories))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
