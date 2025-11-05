using System.Collections;

using UnityEngine;
using UnityEngine.UI;

//참고
//reference1: https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md
//reference2: https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/use-model-output.html
public class VideoMatting : MonoBehaviour
{
    public RenderTexture OutputTexture => ouputCamera?.targetTexture;

    [SerializeField] private RenderTexture sourceTexture; //source texture 에 원본 데이터 넣기
    [SerializeField] private Unity.InferenceEngine.ModelAsset modelAsset;
    [SerializeField] private Material alphaMaterial;
    [SerializeField] private RawImage sketchRawImage;
    [SerializeField] private Camera ouputCamera;
    [SerializeField] private Vector2 frameResolution = new Vector2(1920, 1080);
    [SerializeField] private RawImage debugRawImage;

    private RenderTexture _foregroundTexture;
    private RenderTexture _alphaTexture;
    private Unity.InferenceEngine.Worker _worker;
    private Unity.InferenceEngine.Model _runtimeModel;
    private RenderTexture _resultRenderTexture;
    private Unity.InferenceEngine.Tensor<float> _r1, _r2, _r3, _r4, _inputTensor, _downsampleRatioTensor;
    private Vector2 _previousResolution;

    void Awake()
    {
        //initialize model
        _runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
        _worker = new Unity.InferenceEngine.Worker(_runtimeModel, Unity.InferenceEngine.BackendType.GPUCompute);
        _r1 = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 1, 1, 1), new float[] { 0.0f });
        _r2 = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 1, 1, 1), new float[] { 0.0f });
        _r3 = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 1, 1, 1), new float[] { 0.0f });
        _r4 = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 1, 1, 1), new float[] { 0.0f });
        _inputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 3, 1, 1));
        _downsampleRatioTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1), new float[] { 1.0f });
        ouputCamera.backgroundColor = new Color(0, 0, 0, 0);
        sketchRawImage.material = alphaMaterial;
    }

    void Start()
    {
        StartCoroutine(ProcessVideoMatting());
    }
    
    void Update()
    {
        UpdateResultRenderTexture();
        UpdateDebugRawImage();
    }

    public void SetSourceTexture(RenderTexture sourceTexture)
    {
        this.sourceTexture = sourceTexture;
    }

    void UpdateResultRenderTexture()
    {
        bool changedResolution = _previousResolution.x != frameResolution.x || _previousResolution.y != frameResolution.y;
        GetOrCreateRenderTexture(ref _resultRenderTexture, (int)frameResolution.x, (int)frameResolution.y, "ResultRT", changedResolution);
        if (ouputCamera != null && ouputCamera.targetTexture == null)
            ouputCamera.targetTexture = _resultRenderTexture;
        _previousResolution = frameResolution;
    }

    void UpdateDebugRawImage()
    {
        if(debugRawImage != null && OutputTexture != null)
            debugRawImage.texture = OutputTexture;
    }

    IEnumerator ProcessVideoMatting()
    {
        while (true)
        {
            if (sourceTexture == null)
            {
                yield return null;
                continue;
            }

            int textureWidth = sourceTexture.width;
            int textureHeight = sourceTexture.height;


            float optimalRatio = CalculateOptimalDownsampleRatio(textureWidth, textureHeight); // get downsaple ratio
            var inputShape = new Unity.InferenceEngine.TensorShape(1, 3, textureHeight, textureWidth); // batch, channel, height, width
            if (_inputTensor == null || !_inputTensor.shape.Equals(inputShape))
            {
                _inputTensor?.Dispose();
                _inputTensor = new Unity.InferenceEngine.Tensor<float>(inputShape);
            }
            Unity.InferenceEngine.TextureConverter.ToTensor(sourceTexture, _inputTensor, new Unity.InferenceEngine.TextureTransform());
            _downsampleRatioTensor[0] = optimalRatio;

            _worker.SetInput("src", _inputTensor);
            _worker.SetInput("r1i", _r1);
            _worker.SetInput("r2i", _r2);
            _worker.SetInput("r3i", _r3);
            _worker.SetInput("r4i", _r4);
            _worker.SetInput("downsample_ratio", _downsampleRatioTensor);
            _worker.Schedule();

            yield return null;

            var foregroundTensor = _worker.PeekOutput("fgr") as Unity.InferenceEngine.Tensor<float>;
            var alphaTensor = _worker.PeekOutput("pha") as Unity.InferenceEngine.Tensor<float>;

            GetOrCreateRenderTexture(ref _foregroundTexture, textureWidth, textureHeight, "ForegroundRT");
            GetOrCreateRenderTexture(ref _alphaTexture, textureWidth, textureHeight, "AlphaRT");

            var fgrAwaiter = foregroundTensor.ReadbackAndCloneAsync().GetAwaiter();
            var alphaAwaiter = alphaTensor.ReadbackAndCloneAsync().GetAwaiter();

            while (!fgrAwaiter.IsCompleted || !alphaAwaiter.IsCompleted)
            {
                yield return null;
            }

            using (var foregroundOut = fgrAwaiter.GetResult())
            using (var alphaOut = alphaAwaiter.GetResult())
            {
                Unity.InferenceEngine.TextureConverter.RenderToTexture(foregroundTensor, _foregroundTexture);
                Unity.InferenceEngine.TextureConverter.RenderToTexture(alphaTensor, _alphaTexture);
            }
            
            try
            {
                if(sketchRawImage != null)
                {
                    sketchRawImage.material.SetTexture("_FgrTex", _foregroundTexture);
                    sketchRawImage.material.SetTexture("_PhaTex", _alphaTexture);
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError("NOTE: Please make sure the RawImage has a material using the VideoMatting shader. Exception: " + e.Message);
            }
        }
    }

    private RenderTexture GetOrCreateRenderTexture(ref RenderTexture renderTexture, int width, int height, string name, bool forceCreate = false)
    {
        if (renderTexture == null || renderTexture.width != width || renderTexture.height != height || forceCreate)
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                DestroyImmediate(renderTexture);
            }

            renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            renderTexture.name = name;
            renderTexture.Create();
        }

        return renderTexture;
    }


    // | Resolution    | Portrait      | Full-Body      |
    // | ------------- | ------------- | -------------- |
    // | <= 512x512    | 1             | 1              |
    // | 1280x720      | 0.375         | 0.6            |
    // | 1920x1080     | 0.25          | 0.4            |
    // | 3840x2160     | 0.125         | 0.2            |
    // 5번의 다운샘플링이 모델 내에서 이루어지는데, 아래의 값으로 다운샘플링이 5번 이루어질때 홀수값이 나와선 안됨.
    // width height 변경시 참고
    private float CalculateOptimalDownsampleRatio(int width, int height)
    {
        int imagePixelCount = width * height;

        if (imagePixelCount <= 512 * 512)
        {
            return 1.0f;     // 원본 크기 유지
        }
        else if (imagePixelCount <= 1280 * 720)
        {
            return 0.6f;
        }
        else if (imagePixelCount <= 1920 * 1080)
        {
            return 0.4f;
        }
        else if (imagePixelCount <= 3840 * 2160)
        {
            return 0.2f;
        }
        else
        {
            return 0.1f;
        }
    }

    void OnDestroy()
    {
        _r1?.Dispose();
        _r2?.Dispose();
        _r3?.Dispose();
        _r4?.Dispose();
        _inputTensor?.Dispose();
        _downsampleRatioTensor?.Dispose();
        _worker?.Dispose();
    }
}