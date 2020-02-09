using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Jobs;

namespace NeuralBurst
{
    public class InputLayer : NetworkLayer
    {

        public InputLayer(LayerParamaters layerParamaters)
        {
            InitLayerInput(layerParamaters);
        }


        public override JobHandle EvaluateLayer(EvaluatorLayer lastLayerState, EvaluatorLayer currentLayerState, int count, JobHandle jobHandleToWaitOn)
        {
            //Never needs to be evaluated
            throw new NotImplementedException();
        }

        public override JobHandle BackpropigateLayer(EvaluatorLayer currentLayerState, EvaluatorLayer nextLayerState,int count, JobHandle jobHandleToWaitOn)
        {
            //Never needs to be evaluated ?
            throw new NotImplementedException();
        }
    }
}
