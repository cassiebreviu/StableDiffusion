using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StableDiffusion
{
    public class AzureOpenAiResult
    {
        public string id { get; set; }
        public Result result { get; set; }
        public string status { get; set; }
    }


    public class Result
    {
        public string caption { get; set; }
        public string contentUrl { get; set; }
        public string contentUrlExpiresAt { get; set; }
        public string createdDateTime { get; set; }
    }
}
