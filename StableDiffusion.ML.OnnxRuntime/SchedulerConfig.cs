namespace StableDiffusion.ML.OnnxRuntime
{
    public class SchedulerConfig
    {
        public int TrainTimesteps { get; set; } = 1000;
        public float BetaStart { get; set; } = 0.00085f;
        public float BetaEnd { get; set; } = 0.012f;
        public IEnumerable<float> TrainedBetas { get; set; }
        public SchedulerBetaSchedule BetaSchedule { get; set; } = SchedulerBetaSchedule.ScaledLinear;
    }
}