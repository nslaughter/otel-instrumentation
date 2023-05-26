
# QUESTIONS:
# - what is the rate limit for each model?
# - what is the cost of each model?

# TODO: confirm pricing and rate limits in OpenAI docs
# TODO: wire more restrictive cost calculations specific per endpoint
# TODO: add audio, image, and file endpoints

# computes cost of a request based on model and usage
def calculate_cost(response):
    if response.model in ['gpt-4', 'gpt-4-0314']:
        cost = (response.usage.prompt_tokens * 0.03 + response.usage.completion_tokens * 0.06) / 1000
    elif response.model in ['gpt-4-32k', 'gpt-4-32k-0314']:
        cost = (response.usage.prompt_tokens * 0.06 + response.usage.completion_tokens * 0.12) / 1000
    elif 'gpt-3.5-turbo' in response.model:
        cost = response.usage.total_tokens * 0.002 / 1000
    elif 'text-embedding-ada-002':
        cost = response.usage.total_tokens * 0.002 / 1000
    elif 'davinci' in response.model:
        cost = response.usage.total_tokens * 0.02 / 1000
    elif 'curie' in response.model:
        cost = response.usage.total_tokens * 0.002 / 1000
    elif 'babbage' in response.model:
        cost = response.usage.total_tokens * 0.0005 / 1000
    elif 'ada' in response.model:
        cost = response.usage.total_tokens * 0.0004 / 1000
    else:
        cost = 0
    return cost

# from OpenAI: https://platform.openai.com/docs/guides/rate-limits/overview
# During the limited beta rollout of GPT-4, the model will have more aggressive rate limits
# to keep up with demand. Default rate limits for gpt-4/gpt-4-0314 are 40k TPM and 200 RPM.
# Default rate limits for gpt-4-32k/gpt-4-32k-0314 are 80k TPM and 400 RPM.

def get_rate_limits(model):
    """Returns a rate-limits tuple (requests per minute, tokens per minute)."""
    if 'gpt-4-32k' in model:
        return (400, 80000)
    elif 'gpt-4' in model:
        return (200, 40000)
    elif 'gpt-3.5-turbo' in model:
        return (3500, 90000)
    elif 'embedding' in model:
        return (3500, 350000)
    else:
        return (0, 0)

    # elif 'davinci' in model:
    #     rate_limit_rpm = 50
    # elif 'curie' in model:
    #     rate_limit_rpm = 10
    # elif 'babbage' in model:
    #     rate_limit_rpm = 1
    # elif 'ada' in model:
    #     rate_limit_rpm = 1
    # else:
    #     rate_limit_rpm = 0
    # return rate_limit_rpm
