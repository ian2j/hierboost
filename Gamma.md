We have that for s = sigma + i*t where sigma > 0 and t >= 0:
    
    1. zeta(s) = sum_{n=1}^{infinity} n^{-s} 
    
    2. eta(s) = sum_{n=1}^{infinity} (-1)^{n+1} * n^{-s}
    
Moreover, we know that eta(s) = (1-2^{1-s}) * zeta(s).

eta(s) = 0 trivially infinitely many times when (1-2^{1-s}) = 0.

One way to write the analytic continuation of zeta(s) for Re(s) > 0 is:

    3. mu(s) = limit_{N to infinity} sum_{n=1}^{N} n^{-s} - N^{1-s}/(1-s)
    
If we consider sigma approaching 1 from the right and t approaching 0 from the right, then we have that Re(mu(s)) approaches the Euler-Mascheroni constant of gamma which is approximately 0.577. Because we can also express mu(s) in terms of eta(s), in writing out this limit, we can unite at least 3 of the so-called most important mathematical constants: pi, e, and gamma.

I find this to be a rather fascinating expression, and given the fact that pi and e have already both been proven to be irrational numbers, it makes me curious if there might be a way yet to prove that gamma too is irrational.

This is another open problem in math where I think you might be helpful in possibly crafting together an attack vector. If we assume gamma = a/b for some natural numbers a and b, with b > a, then can we find a way to show a contradiction which proves that this is impossible? 

To be honest upfront, I do not expect you nor me to have an answer to this question immediately. Maybe we might not ever even get close to one. That said, I am hopeful that we can brainstorm together, iterate on this, and possibly discover new math along the way, and maybe, just maybe, eventually getting to an answer. 

Please digest this so far, and then let's start brainstorming.
