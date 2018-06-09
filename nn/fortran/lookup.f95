module lookup
    implicit none

    real(8),external :: ddot

    type lookup_table
        real(8) :: xmin
        real(8) :: xmax
        real(8) :: scl
        real(8) :: add
        real(8),allocatable :: array(:)
    end type lookup_table

    !* number of elements in all univariate lookup tables
    integer,private :: lookup_table_size = 50000

    integer,allocatable,public :: map_to_tbl_idx(:,:)

    !* array of all look up tables in use
    type(lookup_table),private,allocatable :: lookup_tables(:)

    contains
        real(8) function x_from_idx(xmin,xmax,N,idx)
            !* x(idx=1) = xmin , x(idx=N) = xmax

            real(8),intent(in) :: xmin,xmax
            integer,intent(in) :: N,idx

            x_from_idx = xmin + (xmax-xmin)*dble(idx-1)/dble(N-1)
        end function x_from_idx

        subroutine check_ftype_consistent(expected_type,ft)
            use feature_config

            implicit none

            integer,intent(in) :: ft
            character(len=*),intent(in) :: expected_type

            if (featureID_StringToInt(expected_type).ne.feature_params%info(ft)%ftype) then
                call lookup_error("check_ftype_consistent",&
                        &"wrong function type associated with feat.")
            end if
        end subroutine

        real(8) function wrap_cosine(x)
            implicit none

            real(8),intent(in) :: x

            real(8) :: val
            ! make sure cos(theta) = [-1,1]

            if (x.gt.1.0d0) then
                val = 1.0d0 - (x - 1.0d0)
            else if (x.lt.-1.0d0) then
                val = -1.0d0 + (-1.0d0 - x)
            else
                val = x
            end if
            wrap_cosine = val
        end function wrap_cosine

        subroutine fill_lookup(tbl,func_type,ft)
            use feature_config
            use tapering, only : taper_1,taper_deriv_1

            implicit none

            integer,intent(in) :: tbl,ft
            character(len=*),intent(in) :: func_type

            integer :: N,idx,ftype,ww,num_weights
            real(8) :: x,func_val,xi
            real(8) :: max_rcuts(1:3),rs,eta,lambda
            real(8) :: mean_scl,prec_scl,mean_vec(1:3)
            real(8) :: prec_vec(1:3,1:3),fs,rcut,scl
            real(8) :: k_cnst,ww_dble
            real(8),allocatable :: phi(:)
            
            !* basis parameters
            ftype  = feature_params%info(ft)%ftype
            rcut   = feature_params%info(ft)%rcut
            rs     = feature_params%info(ft)%rs
            fs     = feature_params%info(ft)%fs
            eta    = feature_params%info(ft)%eta
            lambda = feature_params%info(ft)%lambda
            scl    = feature_params%info(ft)%scl_cnst
            xi     = feature_params%info(ft)%xi
            if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                prec_scl = feature_params%info(ft)%prec(1,1)
                mean_scl = feature_params%info(ft)%mean(1)
            else if (ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                prec_vec = feature_params%info(ft)%prec
                mean_vec = feature_params%info(ft)%mean
            else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                num_weights = size(feature_params%info(ft)%linear_w)
                if (allocated(phi)) then
                    deallocate(phi)
                end if
                allocate(phi(num_weights))
            end if

            if (.not.allocated(lookup_tables)) then
                write(*,*) 'need to allocate all table mem first'
                call exit(0)
            end if
            if (allocated(lookup_tables(tbl)%array)) then
                deallocate(lookup_tables(tbl)%array)
            end if

            max_rcuts(1) = lookup_maxrcut(0)   ! 2+3 body
            max_rcuts(2) = lookup_maxrcut(1)   ! 2 body
            max_rcuts(3) = lookup_maxrcut(2)   ! 3 body

            if (func_type.eq."sqrt") then
                !* sqrt(x)
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = 4.0d0*max_rcuts(1)**2 ! drjk can be at most 2*rcut
            else if (func_type.eq."acsf_behler-g2_a") then
                !* exp(-eta*(dr-rs)^2) * taper(dr,rcut,fs) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_behler-g2",ft)
            else if (func_type.eq."acsf_behler-g2_b") then
                !* d/dr exp(-eta*(dr-rs)^2) * taper(dr,rcut,fs) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_behler-g2",ft)
            else if (func_type.eq."acsf_normal-b2_a") then
                !* exp(-0.5*prec*(dr-mean)^2) * taper(dr,rcut,fs) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_normal-b2",ft)
            else if (func_type.eq."acsf_normal-b2_b") then
                !* d/dr exp(-0.5*prec*(dr-mean)^2) * taper(dr,rcut,fs) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_normal-b2",ft)
            else if (func_type.eq."acsf_fourier-b2_a") then
                !* taper(dr,rcut,fs) * scl * sum_k sin(2 pi k * dr/rcut)
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_fourier-b2",ft)
            else if (func_type.eq."acsf_fourier-b2_b") then
                !* d/dr taper(dr,rcut,fs) * scl * sum_k sin(2 pi k * dr/rcut)
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = max_rcuts(2)
                
                call check_ftype_consistent("acsf_fourier-b2",ft)
            else if (func_type.eq."acsf_behler-g4_a") then
                !* 2^(1-xi) * exp(-eta*x) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = 3.0d0 * max_rcuts(3)**2 
                
                call check_ftype_consistent("acsf_behler-g4",ft)
            else if (func_type.eq."acsf_behler-g4_b") then
                !* (1+lambda x)^xi
                lookup_tables(tbl)%xmin = -1.0d0
                lookup_tables(tbl)%xmax = 1.0d0
                
                call check_ftype_consistent("acsf_behler-g4",ft)
            else if (func_type.eq."acsf_behler-g4_c") then
                !* (1+lambda cos(x))^(xi-1) 
                lookup_tables(tbl)%xmin = -1.0d0
                lookup_tables(tbl)%xmax = 1.0d0
                
                call check_ftype_consistent("acsf_behler-g4",ft)
            else if (func_type.eq."acsf_behler-g5_a") then
                !* 2^(1-xi) * exp(-eta*x) * scl
                lookup_tables(tbl)%xmin = 0.0d0
                lookup_tables(tbl)%xmax = 2.0d0 * max_rcuts(3)**2 
                
                call check_ftype_consistent("acsf_behler-g5",ft)
            else if (func_type.eq."acsf_behler-g5_b") then
                !* (1+lambda cos(x))^xi
                lookup_tables(tbl)%xmin = -1.0d0
                lookup_tables(tbl)%xmax = 1.0d0
                
                call check_ftype_consistent("acsf_behler-g5",ft)
            else if (func_type.eq."acsf_behler-g5_c") then
                !* (1+lambda cos(x))^(xi-1) 
                lookup_tables(tbl)%xmin = -1.0d0
                lookup_tables(tbl)%xmax = 1.0d0
                
                call check_ftype_consistent("acsf_behler-g5",ft)
            else
                call lookup_error("fill_lookup","unsupported function type")
            end if


            allocate(lookup_tables(tbl)%array(lookup_table_size+1))
            N = lookup_table_size
            
            !* pre-compute some factorisable constants
            lookup_tables(tbl)%scl = dble(N-1)/(lookup_tables(tbl)%xmax-lookup_tables(tbl)%xmin)
            lookup_tables(tbl)%add = 1.0d0 - lookup_tables(tbl)%xmin*dble(N-1)/&
                    &(lookup_tables(tbl)%xmax-lookup_tables(tbl)%xmin)
           

            
            do idx=1,N+1
                !* x = [xmin,xmax]
                x = x_from_idx(lookup_tables(tbl)%xmin,lookup_tables(tbl)%xmax,N,idx)
            
            
                if (func_type.eq."sqrt") then
                    !* sqrt(x)
                    func_val = sqrt(x)
                else if (func_type.eq."acsf_behler-g2_a") then
                    !* exp(-eta*(dr-rs)^2) * taper(dr,rcut,fs) * scl
                    func_val = exp(-eta*(x-rs)**2) * taper_1(x,rcut,fs) * scl
                else if (func_type.eq."acsf_behler-g2_b") then
                    !* d/dr exp(-eta*(dr-rs)^2) * taper(dr,rcut,fs) * scl
                    func_val = exp(-eta*(x-rs)**2) * (taper_deriv_1(x,rcut,fs) - &
                            &2.0d0*eta*(x-rs)*taper_1(x,rcut,fs)) * scl
                else if (func_type.eq."acsf_normal-b2_a") then
                    !* exp(-0.5*prec*(dr-mean)^2) * taper(dr,rcut,fs) * scl
                    func_val = exp(-0.5d0*prec_scl*(x-mean_scl)**2) * taper_1(x,rcut,fs) * scl
                else if (func_type.eq."acsf_normal-b2_b") then
                    !* d/dr exp(-0.5*prec*(dr-mean)^2) * taper(dr,rcut,fs) * scl
                    func_val = exp(-0.5d0*prec_scl*(x-mean_scl)**2) * (taper_deriv_1(x,rcut,fs) - &
                            &prec_scl*(x-mean_scl)*taper_1(x,rcut,fs)) * scl
                else if (func_type.eq."acsf_fourier-b2_a") then
                    !* taper(dr,rcut,fs) * scl * sum_k sin(2 pi k * dr/rcut)
                    k_cnst = 6.28318530718 / rcut
                    do ww=1,num_weights,1
                        ww_dble = dble(ww)
                        phi(ww) = sin(ww_dble * k_cnst * x)
                    end do
                    func_val = ddot(num_weights,feature_params%info(ft)%linear_w,1,phi,1) *&
                            &taper_1(x,rcut,fs)*scl
                else if (func_type.eq."acsf_fourier-b2_b") then
                    !* d/dr taper(dr,rcut,fs) * scl * sum_k sin(2 pi k * dr/rcut)
                    k_cnst = 6.28318530718 / rcut
                    do ww=1,num_weights,1
                        ww_dble = dble(ww)
                        phi(ww) = taper_deriv_1(x,rcut,fs) * sin(ww_dble*k_cnst*x) + &
                                &taper_1(x,rcut,fs) * ww_dble * k_cnst * cos(ww_dble*k_cnst*x)
                    end do
                    func_val = ddot(num_weights,feature_params%info(ft)%linear_w,1,phi,1) * scl
                else if (func_type.eq."acsf_behler-g4_a") then
                    !* 2^(1-xi) * exp(-eta*x) * scl
                    func_val = 2.0d0**(1.0d0-xi) * exp(-eta*x) * scl
                else if (func_type.eq."acsf_behler-g4_b") then
                    !* (1+lambda x)^xi
                    x = wrap_cosine(x)  !* make sure cos(theta) = [-1,1]
                    func_val = (1.0d0 + lambda*x)**xi
                else if (func_type.eq."acsf_behler-g4_c") then
                    !* (1+lambda x))^(xi-1) 
                    x = wrap_cosine(x)  !* make sure cos(theta) = [-1,1]
                    func_val = (1.0d0 + lambda*x)**(xi-1.0d0)
                else if (func_type.eq."acsf_behler-g5_a") then
                    !* 2^(1-xi) * exp(-eta*x) * scl
                    func_val = 2.0d0**(1.0d0-xi) * exp(-eta*x) * scl
                else if (func_type.eq."acsf_behler-g5_b") then
                    !* (1+lambda x)^xi
                    x = wrap_cosine(x)  !* make sure cos(theta) = [-1,1]
                    func_val = (1.0d0+lambda*x)**xi
                else if (func_type.eq."acsf_behler-g5_c") then
                    !* (1+lambda x)^(xi-1) 
                    x = wrap_cosine(x)  !* make sure cos(theta) = [-1,1]
                    func_val = (1.0d0+lambda*x)**(xi-1.0d0)
                else
                    call lookup_error("fill_lookup","unsupported function type")
                end if
               

                !* function value at grid point idx
                lookup_tables(tbl)%array(idx) = func_val
            end do

            if (.not.check_table(tbl)) then
                call lookup_error("fill_lookup","access_lookup inconsistent with stored array")
            end if
    
            if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                deallocate(phi)
            end if
        end subroutine fill_lookup

        real(8) function access_lookup(x,tbl)
            implicit none

            real(8),intent(in) :: x
            integer,intent(in) :: tbl
            
            integer :: int_rounded_idx
            real(8) :: real_idx,real_rounded_idx
            real(8) :: y1,y2

            !* (x - table%xmin)*dble(N-1)/(table%xmax-table%xmin) + 1.0d0
            real_idx = x*lookup_tables(tbl)%scl + lookup_tables(tbl)%add

            ! [1,N] 
            real_rounded_idx = floor(real_idx)

            int_rounded_idx = int(real_rounded_idx)
            y1 = lookup_tables(tbl)%array(int_rounded_idx)
            y2 = lookup_tables(tbl)%array(int_rounded_idx+1)

            access_lookup = (real_idx - real_rounded_idx)*(y2-y1) + y1
        end function access_lookup

        real(8) function func1(x)
            implicit none
            real(8),intent(in) :: x

            func1 = sin(x*6.7d0) + 0.123d0
        end function func1

        logical function check_table(tbl)
            use util, only : scalar_equal
            
            implicit none

            integer,intent(in) :: tbl

            real(8) :: x1,x2,x
            logical :: res
            integer :: N,idx

            res = .true.

            N = lookup_table_size
            if (size(lookup_tables(tbl)%array).ne.N+1) then
                res = .false.
            end if

            !* check first and last x coordinates are OK
            x1 = x_from_idx(lookup_tables(tbl)%xmin,lookup_tables(tbl)%xmax,N,1)
            x2 = x_from_idx(lookup_tables(tbl)%xmin,lookup_tables(tbl)%xmax,N,N)
            if ((abs(x1-lookup_tables(tbl)%xmin).gt.1e-15).or.&
            &(abs(x2-lookup_tables(tbl)%xmax).gt.1e-15)) then
                res = .false.
            end if
           
            !* check function value at grid points is consistent
            do idx=1,N
                !* x = [xmin,xmax]
                x = x_from_idx(lookup_tables(tbl)%xmin,lookup_tables(tbl)%xmax,N,idx)

                if(.not.scalar_equal(access_lookup(x,tbl),lookup_tables(tbl)%array(idx),&
                &dble(1e-10),dble(1e-30),.false.)) then    
                    res = .false.
                end if
            end do
            
            if (.not.res) then
            write(*,*) 'BAD TBL:',tbl,lookup_tables(tbl)%array(1:2)
            end if

            check_table = res
        end function check_table

        integer function num_tables_for_feature(ft)
            !* return number of univariate lookup tables needed for given feature
            use feature_config
            
            implicit none

            !* args
            integer,intent(in) :: ft

            !* scratch
            integer :: ftype,num_tables

            !* default value
            num_tables = 0

            ftype = feature_params%info(ft)%ftype

            if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                num_tables = 2
            else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                num_tables = 2
            else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                num_tables = 2
            else if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                num_tables = 3
            else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                num_tables = 3
            end if

            num_tables_for_feature = num_tables
        end function num_tables_for_feature

        integer function max_num_tables_per_feature()
            use feature_config
            
            implicit none

            integer :: ft,max_num

            max_num = 0
            do ft=1,feature_params%num_features
                if (num_tables_for_feature(ft).gt.max_num) then
                    max_num = num_tables_for_feature(ft)
                end if
            end do
            max_num_tables_per_feature = max_num
        end function max_num_tables_per_feature

        integer function total_num_tables()
            use feature_config
            
            implicit none

            integer :: cntr,ft

            !* always use lookup for sqrt(dr)
            cntr = 1

            do ft=1,feature_params%num_features,1
                cntr = cntr + num_tables_for_feature(ft)
            end do

            total_num_tables = cntr
        end function total_num_tables

        subroutine init_lookup_tables()
            use feature_config
            
            implicit none

            integer :: N,table,ft,ftype,tmp

            if (allocated(lookup_tables)) then
                deallocate(lookup_tables)
            end if

            N = total_num_tables()
            allocate(lookup_tables(N))
      
            !* go from (equation number,feature idx) to tble idx
            allocate(map_to_tbl_idx(max_num_tables_per_feature(),feature_params%num_features))
            map_to_tbl_idx = -1

            !* this order is always true
            call fill_lookup(1,"sqrt",1)

            table = 2
            do ft=1,feature_params%num_features,1
                tmp = num_tables_for_feature(ft)
                if (tmp.le.0) then
                    cycle
                end if

                ftype = feature_params%info(ft)%ftype
                
                if (ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    !* create arrays
                    call fill_lookup(table,"acsf_behler-g2_a",ft)
                    call fill_lookup(table+1,"acsf_behler-g2_b",ft)
                    
                    !* map back to table index
                    map_to_tbl_idx(1,ft) = table
                    map_to_tbl_idx(2,ft) = table + 1
                else if (ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    !* create arrays
                    call fill_lookup(table,"acsf_normal-b2_a",ft)
                    call fill_lookup(table+1,"acsf_normal-b2_b",ft)
                    
                    !* map back to table index
                    map_to_tbl_idx(1,ft) = table
                    map_to_tbl_idx(2,ft) = table + 1
                else if (ftype.eq.featureID_StringToInt("acsf_fourier-b2")) then
                    !* create arrays
                    call fill_lookup(table,"acsf_fourier-b2_a",ft)
                    call fill_lookup(table+1,"acsf_fourier-b2_b",ft)
                    
                    !* map back to table index
                    map_to_tbl_idx(1,ft) = table
                    map_to_tbl_idx(2,ft) = table + 1
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    !* create arrays
                    call fill_lookup(table,"acsf_behler-g4_a",ft)
                    call fill_lookup(table+1,"acsf_behler-g4_b",ft)
                    call fill_lookup(table+2,"acsf_behler-g4_c",ft)
                    
                    !* map back to table index
                    map_to_tbl_idx(1,ft) = table
                    map_to_tbl_idx(2,ft) = table + 1
                    map_to_tbl_idx(3,ft) = table + 2
                else if (ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    !* create arrays
                    call fill_lookup(table,"acsf_behler-g5_a",ft)
                    call fill_lookup(table+1,"acsf_behler-g5_b",ft)
                    call fill_lookup(table+2,"acsf_behler-g5_c",ft)
                    
                    !* map back to table index
                    map_to_tbl_idx(1,ft) = table
                    map_to_tbl_idx(2,ft) = table + 1
                    map_to_tbl_idx(3,ft) = table + 2
                end if

                !* book keeping
                table = table + tmp
            end do
        end subroutine init_lookup_tables

        subroutine lookup_error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message
            character,dimension(1:len(routine)+26) :: header
            header(:) = "*"

            write(*,*) ''
            write(*,*) header
            write(*,*) 'error raised in routine :',routine
            write(*,*) header
            write(*,*) ''
            write(*,*) 'Error : ',message
            call exit(0)
        end subroutine lookup_error
        
        real(8) function lookup_maxrcut(arg)
            !===============================================================!
            ! Return maximum cut off radius of all current features         !
            !                                                               !
            ! Input                                                         !
            ! -----                                                         !
            !   - arg : 0 = max of all features                             !
            !           1 = max of all isotropic features                   !
            !           2 = max of all anisotropic features                 !
            !===============================================================!
            use feature_config
            
            implicit none

            integer,intent(in) :: arg

            real(8),allocatable :: tmprcut(:)
            integer :: ii,ftype
            real(8) :: tmpr

            allocate(tmprcut(feature_params%num_features))

            do ii=1,feature_params%num_features,1
                !* feature type
                ftype = feature_params%info(ii)%ftype

                !* interaction cut off (can be null)
                tmpr = feature_params%info(ii)%rcut

                if (arg.eq.0) then
                    if (ftype.eq.featureID_StringToInt("atomic_number")) then
                        tmprcut(ii) = -1.0d0
                    else
                        !* all features
                        tmprcut(ii) = tmpr
                    end if
                else if (arg.eq.1) then
                    if (feature_IsTwoBody(ftype)) then
                        tmprcut(ii) = tmpr
                    else
                        tmprcut(ii) = -1.0d0
                    end if
                else if (arg.eq.2) then
                    if ( (feature_IsTwoBody(ftype).neqv..true.).and.(ftype.ne.&
                    &featureID_StringToInt("atomic_number")) ) then
                        tmprcut(ii) = tmpr
                    else
                        tmprcut(ii) = -1.0d0
                    end if
                end if
            end do

            tmpr = maxval(tmprcut)
            deallocate(tmprcut)
    
            lookup_maxrcut = tmpr
        end function lookup_maxrcut
end module lookup
