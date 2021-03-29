function FEVAL = NC_NormalizedObjFunc(x, realFunction, pyContext, realBounds)
    global numCalls;
    if size(x,2)~=1 || size(x,1)~=size(realBounds,1)
       error('Bad size of the input given to the normalized objective function'); 
    end
    realX = x .* (realBounds(:,2) - realBounds(:,1)) + realBounds(:,1); % val*(max-min) + min
    FEVAL = realFunction(realX, pyContext);
    numCalls = numCalls + 1;
end
